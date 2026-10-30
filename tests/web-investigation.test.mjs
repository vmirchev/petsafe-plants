import assert from "node:assert/strict";
import test from "node:test";
import {
  createInvestigationState,
  largestSpecimenIds,
  readPipelineEvents,
} from "../web/investigation.mjs";

function streamText(text, chunkSize = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(controller) {
      for (let start = 0; start < bytes.length; start += chunkSize) {
        controller.enqueue(bytes.slice(start, start + chunkSize));
      }
      controller.close();
    },
  });
}

async function events(text, chunkSize) {
  const result = [];
  for await (const event of readPipelineEvents(streamText(text, chunkSize)))
    result.push(event);
  return result;
}

test("single plant / skipped SAM completes directly from the scene assessment", () => {
  const machine = createInvestigationState();
  machine.transition("uploading");
  assert.equal(machine.busy, true);
  machine.transition("analyzing_scene");
  assert.equal(machine.busy, true);
  machine.transition("scene_analyzed");
  machine.transition("completed");
  assert.equal(machine.busy, false);
});

test("SAM flow waits for real masks before investigating regions", () => {
  const machine = createInvestigationState();
  for (const stage of [
    "analyzing_scene",
    "scene_analyzed",
    "segmenting",
    "segments_ready",
    "identifying_regions",
    "completed",
  ]) {
    machine.transition(stage);
    assert.equal(machine.state, stage);
  }
});

test("the first three specimen cards use actual SAM mask area", () => {
  const regions = [
    { object_id: "small", mask_pixels: 120, bounds: [0, 0, 100, 100] },
    { object_id: "large", mask_pixels: 800, bounds: [0, 0, 30, 30] },
    { object_id: "middle", mask_pixels: 400, bounds: [0, 0, 40, 40] },
    { object_id: "equal", mask_pixels: 400, bounds: [0, 0, 50, 50] },
  ];
  assert.deepEqual(largestSpecimenIds(regions), ["large", "middle", "equal"]);
  assert.deepEqual(largestSpecimenIds(regions.slice(0, 2)), ["large", "small"]);
  assert.deepEqual(largestSpecimenIds([{ object_id: 7, bounds: [0, 0, 5, 6] }]), ["7"]);
});

test("empty masks or failed SAM can finish with the original scene report", () => {
  for (const lastStage of ["segmenting", "segments_ready"]) {
    const machine = createInvestigationState();
    for (const stage of ["analyzing_scene", "scene_analyzed", "segmenting"])
      machine.transition(stage);
    if (lastStage === "segments_ready") machine.transition(lastStage);
    machine.transition("completed");
    assert.equal(machine.busy, false);
  }
});

test("received results can enter a presentation pause without changing model stages", () => {
  for (const path of [
    ["analyzing_scene", "scene_analyzed"],
    ["analyzing_scene", "scene_analyzed", "segmenting", "segments_ready", "identifying_regions"],
  ]) {
    const machine = createInvestigationState();
    path.forEach((stage) => machine.transition(stage));
    machine.transition("presenting");
    assert.equal(machine.busy, true);
    machine.transition("completed");
    assert.equal(machine.busy, false);
  }
});

test("cancellation, retry, and follow-up can start a fresh investigation", () => {
  const machine = createInvestigationState();
  for (const stage of [
    "analyzing_scene",
    "idle",
    "uploading",
    "analyzing_scene",
    "error",
    "analyzing_scene",
    "scene_analyzed",
    "completed",
    "uploading",
  ]) {
    machine.transition(stage);
  }
  assert.equal(machine.state, "uploading");
});

test("there is no timed state advancement or fabricated segmentation stage", () => {
  const machine = createInvestigationState();
  machine.transition("analyzing_scene");
  machine.transition("analyzing_scene"); // server acknowledgment may repeat the request state
  assert.throws(() => machine.transition("identifying_regions"), /Unexpected/);
  assert.throws(() => machine.transition("50_percent"), /Unknown/);
  assert.equal(machine.state, "analyzing_scene");
});

test("SSE decodes Unicode and CRLF split at every byte; heartbeats do not advance state", async () => {
  const result = await events(
    ': keepalive\r\n\r\nevent: scene_analyzed\r\ndata: {"name":"Orchidée 🌿"}\r\n\r\n',
  );
  assert.deepEqual(result, [
    { event: "scene_analyzed", data: { name: "Orchidée 🌿" } },
  ]);
});

test("SSE preserves event order when multiple events arrive in one chunk", async () => {
  const text = ["analyzing_scene", "scene_analyzed", "completed"]
    .map((event) => `event: ${event}\ndata: {}\n\n`)
    .join("");
  assert.deepEqual(
    (await events(text, 1000)).map((item) => item.event),
    ["analyzing_scene", "scene_analyzed", "completed"],
  );
});

test("SSE multiline data and server errors remain parseable", async () => {
  assert.deepEqual(
    await events('event: error\ndata: {\ndata: "message": "Unavailable"}\n\n'),
    [{ event: "error", data: { message: "Unavailable" } }],
  );
});

test("truncated or malformed SSE fails instead of inventing a completed report", async () => {
  await assert.rejects(
    events('event: completed\ndata: {"value":'),
    /interrupted/,
  );
  await assert.rejects(
    events("event: completed\ndata: NOT_JSON\n\n"),
    SyntaxError,
  );
});

test("leaving a stale stream cancels the reader and releases its lock", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode("event: scene_analyzed\ndata: {}\n\n"),
      );
    },
    cancel() {
      cancelled = true;
    },
  });
  for await (const _event of readPipelineEvents(stream)) break;
  assert.equal(cancelled, true);
  assert.equal(stream.locked, false);
});

test("network failure propagates without leaving the stream locked", async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.error(new Error("Connection lost"));
    },
  });
  await assert.rejects(async () => {
    for await (const _event of readPipelineEvents(stream)) {
    }
  }, /Connection lost/);
  assert.equal(stream.locked, false);
});
