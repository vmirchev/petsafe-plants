// UI state follows real pipeline events, never elapsed inference time.
const transitions = {
  idle: ["uploading", "analyzing_scene"],
  uploading: ["analyzing_scene"],
  analyzing_scene: ["scene_analyzed"],
  scene_analyzed: ["segmenting", "presenting", "completed"],
  segmenting: ["segments_ready", "presenting", "completed"],
  segments_ready: ["identifying_regions", "presenting", "completed"],
  identifying_regions: ["presenting", "completed"],
  presenting: ["completed"],
  completed: ["uploading", "analyzing_scene"],
  error: ["uploading", "analyzing_scene"],
};

export function createInvestigationState() {
  let state = "idle";
  return {
    get state() {
      return state;
    },
    get busy() {
      return !["idle", "completed", "error"].includes(state);
    },
    transition(next) {
      if (!(next in transitions))
        throw new Error(`Unknown investigation state: ${next}`);
      // Upload/removal cancels any stage; a request failure can occur at any stage.
      if (
        next !== state &&
        !["idle", "error"].includes(next) &&
        !transitions[state].includes(next)
      ) {
        throw new Error(
          `Unexpected investigation transition: ${state} → ${next}`,
        );
      }
      state = next;
    },
  };
}

/** Choose the largest selected SAM masks, preserving event order for equal areas. */
export function largestSpecimenIds(regions, limit = 3) {
  const area = (region) => {
    if (Number.isFinite(region.mask_pixels) && region.mask_pixels >= 0)
      return region.mask_pixels;
    const [left, top, right, bottom] = region.bounds || [0, 0, 0, 0];
    return Math.max(0, right - left) * Math.max(0, bottom - top);
  };
  return regions.map((region, index) => ({ region, index }))
    .sort((a, b) => area(b.region) - area(a.region) || a.index - b.index)
    .slice(0, limit)
    .map(({ region }) => String(region.object_id));
}

/** Decode SSE across arbitrary network chunk boundaries; ignore heartbeats. */
export async function* readPipelineEvents(stream) {
  if (!stream)
    throw new Error("This browser did not provide a streaming response.");
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      // Keep CRLF intact across chunk boundaries before finding complete events.
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const block = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        const lines = block.split(/\r?\n/);
        const event = lines
          .find((line) => line.startsWith("event:"))
          ?.slice(6)
          .trim();
        const data = lines
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (event && data) yield { event, data: JSON.parse(data) };
      }
      if (done) {
        if (buffer.trim() && !buffer.trimStart().startsWith(":")) {
          throw new Error("The server response was interrupted. Please retry.");
        }
        break;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
