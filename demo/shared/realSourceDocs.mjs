const PILLARS = [
  {
    concept: "stoic composure under pressure",
    motifs: ["negative visualization", "preparation", "equanimity", "discipline"],
  },
  {
    concept: "clear thinking during leadership stress",
    motifs: ["triage", "signal over noise", "calm delegation", "tempo control"],
  },
  {
    concept: "breath and body control during conflict",
    motifs: ["long exhale", "down-regulation", "shoulder release", "steady gaze"],
  },
  {
    concept: "training attention before chaos arrives",
    motifs: ["rehearsal", "drills", "journaling", "reflection"],
  },
  {
    concept: "resilience through deliberate practice",
    motifs: ["repetition", "small failures", "recovery", "confidence"],
  },
];

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function paragraph(topic, motifs, docIndex, paragraphIndex, rng) {
  const phrases = [
    `The operator notes that staying calm under pressure is not a personality trait but a trained sequence.`,
    `Each rehearsal in document ${docIndex} starts with a pressure cue, a physical reset, and a leadership decision.`,
    `The archive ties ${topic} to practical habits that can be repeated before meetings, incidents, and conflict.`,
    `When a team leader feels urgency rise, the text recommends widening attention before choosing the next action.`,
    `The notes describe how breathing, posture, and language control the speed at which panic spreads through a room.`,
    `Several case studies insist that clarity under pressure comes from preloaded routines rather than inspiration.`,
  ];

  const lines = [];
  for (let i = 0; i < 22; i++) {
    const motif = motifs[(paragraphIndex + i) % motifs.length];
    const phrase = phrases[Math.floor(rng() * phrases.length)];
    lines.push(
      `${phrase} It links ${motif} to calm leadership under pressure, deliberate focus, and resilient decision making.`,
    );
  }

  return lines.join(" ");
}

export function buildRealSourceDocuments() {
  const docs = [];

  for (let docIndex = 0; docIndex < 120; docIndex++) {
    const pillar = PILLARS[docIndex % PILLARS.length];
    const rng = mulberry32(docIndex + 11);
    const paragraphs = [];

    for (let paragraphIndex = 0; paragraphIndex < 8; paragraphIndex++) {
      paragraphs.push(
        paragraph(
          pillar.concept,
          pillar.motifs,
          docIndex,
          paragraphIndex,
          rng,
        ),
      );
    }

    docs.push({
      id: `real-doc-${docIndex}`,
      title: `Pressure Playbook ${docIndex}`,
      text: [
        `Document ${docIndex}: ${pillar.concept}.`,
        `This document is about how to stay calm under pressure while leading a team, making decisions, and controlling attention in uncertain moments.`,
        ...paragraphs,
      ].join("\n\n"),
      metadata: {
        pillar: pillar.concept,
      },
    });
  }

  return docs;
}
