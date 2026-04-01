// Tool definitions for OpenRouter (OpenAI-compatible function calling)

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string; // JSON string
  };
}

export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: 'function';
  function?: {
    name?: string;
    arguments?: string;
  };
}

export interface ToolMessage {
  role: 'tool';
  tool_call_id: string;
  content: string;
}

export interface AssistantToolMessage {
  role: 'assistant';
  content: string | null;
  tool_calls: ToolCall[];
}

/** Models known to support tool/function calling on OpenRouter free tier. */
export const TOOL_CAPABLE_MODELS = new Set([
  // Auto-routers (smart-filter for tool support when tools param is passed)
  'openrouter/auto',
  'openrouter/free',
  // Z.ai / Zhipu
  'z-ai/glm-4.5-air:free',
  // Meta
  'meta-llama/llama-3.3-70b-instruct:free',
  // Qwen
  'qwen/qwen3-coder:free',
  'qwen/qwen3.6-plus-preview:free',
  'qwen/qwen3-next-80b-a3b-instruct:free',
  // OpenAI open-source
  'openai/gpt-oss-120b:free',
  'openai/gpt-oss-20b:free',
  // NVIDIA
  'nvidia/nemotron-3-super-120b-a12b:free',
  'nvidia/nemotron-3-nano-30b-a3b:free',
  'nvidia/nemotron-nano-9b-v2:free',
  // StepFun
  'stepfun/step-3.5-flash:free',
  // Others
  'minimax/minimax-m2.5:free',
  'arcee-ai/trinity-mini:free',
  'arcee-ai/trinity-large-preview:free',
]);

/** OpenAI-compatible tool definitions sent with the chat request. */
export const TOOL_DEFINITIONS = [
  {
    type: 'function' as const,
    function: {
      name: 'run_inference',
      description:
        'Run neutron reflectivity inference. For real experimental data, uses the pre-trained model to predict SLD from uploaded NR. For synthetic data, requires layer parameters.',
      parameters: {
        type: 'object',
        properties: {
          dataSource: {
            type: 'string',
            enum: ['real', 'synthetic'],
            description:
              'Use "real" when the user has uploaded experimental NR data and wants to fit it. Use "synthetic" when generating from layer parameters.',
          },
          layers: {
            type: 'array',
            description:
              'Film layers for synthetic generation. Not needed for real data inference.',
            items: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                sld: { type: 'number', description: 'Scattering Length Density' },
                thickness: { type: 'number', description: 'Thickness in Angstroms' },
                roughness: { type: 'number', description: 'Roughness in Angstroms' },
              },
              required: ['name', 'sld', 'thickness', 'roughness'],
            },
          },
        },
        required: ['dataSource'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'get_backend_status',
      description:
        'Check backend status: which files are uploaded (NR data, SLD data, models, normalization stats), whether pyreflect is available, and current settings. Use this when the user asks about their files or system readiness.',
      parameters: {
        type: 'object',
        properties: {},
      },
    },
  },
];