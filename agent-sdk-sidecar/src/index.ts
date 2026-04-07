import http from "node:http";
import {
  unstable_v2_createSession,
  type SDKSession,
  type SDKMessage,
  type SDKSessionOptions,
} from "@anthropic-ai/claude-agent-sdk";

const PORT = parseInt(process.env.AGENT_SDK_SIDECAR_PORT || "8377", 10);

// Track active sessions by a conversation key
const sessions = new Map<string, SDKSession>();

interface AnthropicRequest {
  model: string;
  messages: Array<{
    role: string;
    content: string | Array<{ type: string; [key: string]: unknown }>;
  }>;
  system?: string | Array<{ type: string; text: string }>;
  tools?: Array<{ name: string; description: string; input_schema: unknown }>;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  tool_choice?: unknown;
  thinking?: unknown;
  output_config?: unknown;
  output_format?: unknown;
  metadata?: unknown;
}

/**
 * Extract the latest user message from the full messages array.
 * Forgecode sends the COMPLETE conversation history each turn.
 * We need to find what's new since the last turn.
 */
function extractLatestUserContent(messages: AnthropicRequest["messages"]): string {
  // Walk backwards to find the latest user message or tool_result
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        return msg.content;
      }
      // Handle content blocks (text, tool_result, etc.)
      const parts: string[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          parts.push(block.text as string);
        } else if (block.type === "tool_result") {
          parts.push(
            `[Tool result for ${block.tool_use_id}: ${
              typeof block.content === "string"
                ? block.content
                : JSON.stringify(block.content)
            }]`
          );
        }
      }
      if (parts.length > 0) return parts.join("\n");
    }
  }
  return "";
}

/**
 * Extract system prompt from the request.
 */
function extractSystemPrompt(req: AnthropicRequest): string | undefined {
  if (!req.system) return undefined;
  if (typeof req.system === "string") return req.system;
  return req.system.map((s) => s.text).join("\n\n");
}

/**
 * Write an SSE event to the response.
 */
function writeSSE(res: http.ServerResponse, eventType: string, data: unknown): void {
  res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
}

/**
 * Handle a POST /v1/messages request by bridging to the Agent SDK V2.
 */
async function handleMessages(
  req: http.IncomingMessage,
  res: http.ServerResponse
): Promise<void> {
  // Read request body
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(chunk as Buffer);
  }
  const body: AnthropicRequest = JSON.parse(Buffer.concat(chunks).toString());

  const latestMessage = extractLatestUserContent(body.messages);
  if (!latestMessage) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "No user message found" }));
    return;
  }

  // Set up SSE response
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  try {
    // Create a new session per request (stateless approach)
    // The Agent SDK handles auth via stored Claude Code CLI credentials
    const sessionOptions: SDKSessionOptions = {
      model: body.model,
      permissionMode: "plan", // Don't execute any tools
      disallowedTools: ["*"], // Remove all built-in tools
    };

    const systemPrompt = extractSystemPrompt(body);
    if (systemPrompt) {
      (sessionOptions as Record<string, unknown>).systemPrompt = systemPrompt;
    }

    const session = unstable_v2_createSession(sessionOptions);

    // Send the latest user message
    await session.send(latestMessage);

    // Stream response events
    for await (const msg of session.stream()) {
      switch (msg.type) {
        case "stream_event": {
          // SDKPartialAssistantMessage — contains raw BetaRawMessageStreamEvent
          // Forward directly as SSE (same format as Anthropic API)
          const streamMsg = msg as unknown as { event: { type: string } };
          const eventType = streamMsg.event.type;
          writeSSE(res, eventType, streamMsg.event);
          break;
        }

        case "assistant": {
          // Full assistant message — the turn is done
          // The stream_events above should have already sent everything
          // but send a final message_stop if needed
          const assistantMsg = msg as {
            message?: { usage?: unknown; stop_reason?: string };
          };
          if (assistantMsg.message) {
            writeSSE(res, "message_delta", {
              type: "message_delta",
              delta: {
                stop_reason: assistantMsg.message.stop_reason || "end_turn",
              },
              usage: assistantMsg.message.usage || {},
            });
            writeSSE(res, "message_stop", { type: "message_stop" });
          }
          break;
        }

        case "result": {
          // Final result — session is done
          const resultMsg = msg as { subtype: string; result?: unknown };
          if (resultMsg.subtype === "error") {
            writeSSE(res, "error", {
              type: "error",
              error: { type: "api_error", message: String(resultMsg.result) },
            });
          }
          break;
        }

        case "system": {
          // Status updates — skip for now
          break;
        }

        default:
          // Other message types — log but don't forward
          console.error(`[sidecar] unhandled message type: ${msg.type}`);
          break;
      }
    }

    // Clean up session
    session.close();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[sidecar] error: ${message}`);
    writeSSE(res, "error", {
      type: "error",
      error: { type: "api_error", message },
    });
  }

  res.end();
}

/**
 * Health check endpoint.
 */
function handleHealth(res: http.ServerResponse): void {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ status: "ok" }));
}

// Create HTTP server
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://localhost:${PORT}`);

  if (req.method === "GET" && url.pathname === "/health") {
    handleHealth(res);
    return;
  }

  if (req.method === "POST" && url.pathname === "/v1/messages") {
    await handleMessages(req, res);
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not found" }));
});

server.listen(PORT, "127.0.0.1", () => {
  // Print port to stdout so the parent process can detect readiness
  console.log(`SIDECAR_READY:${PORT}`);
  console.error(`[sidecar] Agent SDK sidecar listening on http://127.0.0.1:${PORT}`);
});

// Graceful shutdown
process.on("SIGTERM", () => {
  console.error("[sidecar] shutting down...");
  for (const [, session] of sessions) {
    session.close();
  }
  server.close();
  process.exit(0);
});

process.on("SIGINT", () => {
  process.emit("SIGTERM" as never);
});
