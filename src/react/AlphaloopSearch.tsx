"use client";

import { useMemo } from "react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { AssistantChatTransport } from "@assistant-ui/react-ai-sdk";

export interface AlphaloopSearchProps {
  apiUrl?: string;
}

/**
 * Hook that creates an assistant-ui runtime connected to an alphaloop handler.
 */
export function useAlphaloopRuntime({
  apiUrl = "/api/search",
}: { apiUrl?: string } = {}) {
  const transport = useMemo(
    () => new AssistantChatTransport({ api: apiUrl }),
    [apiUrl],
  );
  return useChatRuntime({ transport });
}

// No longer exporting makeAssistantToolUI registration since
// the new components (SearchProgress, Citations) are standalone.
export const DeepSearchToolUIRegistration = () => null;
