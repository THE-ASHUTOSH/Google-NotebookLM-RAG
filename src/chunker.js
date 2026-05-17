/**
 * Recursive character chunking.
 *
 * Strategy: split on the strongest semantic boundary that keeps each piece
 * under `chunkSize`. Order of separators (strongest first):
 *     paragraph break -> line break -> sentence end -> clause -> word -> char
 *
 * A sliding `chunkOverlap` is added between adjacent chunks so a sentence
 * straddling a boundary still appears (in part) on both sides. This preserves
 * local context for retrieval and reduces "lost-at-the-seam" failures.
 */

const DEFAULT_SEPARATORS = ['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ', ''];

/** Recursively split `text` so each piece is <= chunkSize when possible. */
function splitRecursive(text, separators, chunkSize) {
  if (text.length <= chunkSize) return text ? [text] : [];

  const sep = separators[0];
  const rest = separators.slice(1);

  if (sep === '') {
    // Last resort: hard split by character.
    const out = [];
    for (let i = 0; i < text.length; i += chunkSize) out.push(text.slice(i, i + chunkSize));
    return out;
  }

  const parts = text.split(sep);
  const pieces = [];
  for (const part of parts) {
    if (!part) continue;
    if (part.length <= chunkSize) pieces.push(part);
    else pieces.push(...splitRecursive(part, rest, chunkSize));
  }

  // Greedily merge adjacent pieces (re-attaching the separator) while staying
  // under chunkSize, so we don't end up with hundreds of tiny fragments.
  const merged = [];
  let current = '';
  for (const piece of pieces) {
    const candidate = current ? current + sep + piece : piece;
    if (candidate.length <= chunkSize) {
      current = candidate;
    } else {
      if (current) merged.push(current);
      current = piece;
    }
  }
  if (current) merged.push(current);
  return merged;
}

/**
 * Split `text` into overlapping chunks suitable for embedding.
 * Returns objects: { text, index, charStart, charEnd }.
 */
export function chunkText(text, { chunkSize = 1000, chunkOverlap = 150 } = {}) {
  if (chunkOverlap >= chunkSize) throw new Error('chunkOverlap must be smaller than chunkSize');

  text = text.trim();
  if (!text) return [];

  const raw = splitRecursive(text, DEFAULT_SEPARATORS, chunkSize);

  const chunks = [];
  let cursor = 0; // position in original text, for accurate char offsets
  let prevTail = '';

  raw.forEach((piece, i) => {
    const body = prevTail ? `${prevTail} ${piece}`.trim() : piece;
    // Locate the piece in the original text from `cursor` onward so offsets stay
    // accurate even though separators were stripped during splitting.
    let found = text.indexOf(piece, cursor);
    if (found === -1) found = cursor;
    const charStart = Math.max(0, found - prevTail.length);
    const charEnd = found + piece.length;
    cursor = charEnd;

    chunks.push({ text: body, index: i, charStart, charEnd });

    prevTail = chunkOverlap > 0 ? piece.slice(-chunkOverlap) : '';
  });

  return chunks;
}
