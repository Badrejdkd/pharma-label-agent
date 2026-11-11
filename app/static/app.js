// Interface minimale : envoie la question à /ask/stream, affiche chaque étape
// de l'agent en direct, puis la réponse citée et ses sources.
// Aucune dépendance : le texte du modèle est échappé avant tout rendu HTML.

const $ = (id) => document.getElementById(id);

const els = {
  status: $("status"),
  form: $("ask-form"),
  question: $("question"),
  examples: $("examples"),
  button: $("ask-button"),
  result: $("result"),
  progress: $("progress"),
  answer: $("answer"),
  verdict: $("verdict"),
  answerBody: $("answer-body"),
  flags: $("flags"),
  flagsList: $("flags-list"),
  disclaimer: $("disclaimer"),
  trace: $("trace"),
  traceBody: $("trace-body"),
  error: $("error"),
  sources: $("sources"),
  sourcesList: $("sources-list"),
};

const VERDICTS = {
  pass: "Every claim was checked against its source.",
  revised: "Checked against the sources and corrected once.",
  flagged: "Some claims could not be verified against the sources.",
};

// ---------------------------------------------------------------- utilities
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function documentLabel(file) {
  if (file === "GLOSSARY.md") return "Glossary";
  const [source, drug] = file.replace(/\.md$/, "").split("-");
  const name = drug.charAt(0).toUpperCase() + drug.slice(1);
  return source === "dailymed" ? `${name}, FDA label` : `${name}, patient leaflet (MedlinePlus)`;
}

function shortArguments(args) {
  return Object.values(args)
    .filter((value) => value !== null && value !== "")
    .map((value) => (Array.isArray(value) ? value.join(", ") : String(value)))
    .map((value) => (value.length > 60 ? `${value.slice(0, 57)}…` : value))
    .join("; ");
}

// ------------------------------------------------------- markdown (subset)
function inline(escaped) {
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-cite="$1" aria-label="Show source $1">$1</button>');
}

function renderMarkdown(markdown) {
  const html = [];
  let paragraph = [];
  let list = null;
  let table = null;

  const flushParagraph = () => {
    if (paragraph.length) html.push(`<p>${inline(escapeHtml(paragraph.join(" ")))}</p>`);
    paragraph = [];
  };
  const flushList = () => {
    if (list) {
      const items = list.items.map((item) => `<li>${inline(escapeHtml(item))}</li>`).join("");
      html.push(`<${list.type}>${items}</${list.type}>`);
    }
    list = null;
  };
  const flushTable = () => {
    if (table) {
      const [head, ...rows] = table;
      const cells = (row, tag) => row.map((cell) => `<${tag}>${inline(escapeHtml(cell))}</${tag}>`).join("");
      html.push(
        `<table><thead><tr>${cells(head, "th")}</tr></thead><tbody>` +
          rows.map((row) => `<tr>${cells(row, "td")}</tr>`).join("") +
          "</tbody></table>",
      );
    }
    table = null;
  };

  for (const raw of markdown.replace(/\r\n/g, "\n").split("\n")) {
    const line = raw.replace(/\s+$/, "");
    let match;
    if (!line.trim()) {
      flushParagraph();
      flushList();
      flushTable();
    } else if (/^\s*\|.*\|$/.test(line)) {
      flushParagraph();
      flushList();
      const cells = line.trim().slice(1, -1).split("|").map((cell) => cell.trim());
      if (!cells.every((cell) => /^:?-{3,}:?$/.test(cell))) (table ??= []).push(cells);
    } else if ((match = line.match(/^(#{1,4})\s+(.*)$/))) {
      flushParagraph();
      flushList();
      flushTable();
      const level = Math.min(match[1].length + 2, 6);
      html.push(`<h${level}>${inline(escapeHtml(match[2]))}</h${level}>`);
    } else if ((match = line.match(/^\s{0,3}([-*•]|\d+[.)])\s+(.*)$/))) {
      flushParagraph();
      flushTable();
      const type = /\d/.test(match[1]) ? "ol" : "ul";
      if (!list || list.type !== type) {
        flushList();
        list = { type, items: [] };
      }
      list.items.push(match[2]);
    } else if (list && /^\s{2,}\S/.test(raw)) {
      list.items[list.items.length - 1] += ` ${line.trim()}`; // suite de l'élément de liste
    } else {
      flushList();
      flushTable();
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  flushList();
  flushTable();
  return html.join("");
}

// ----------------------------------------------------------- live progress
let pending = null;

function setPending(text) {
  if (pending) pending.remove();
  pending = el("li", text, "is-current");
  els.progress.append(pending);
}

function addDone(text) {
  const item = el("li", text);
  if (pending) els.progress.insertBefore(item, pending);
  else els.progress.append(item);
}

function onStep({ node, detail }) {
  if (node === "plan") {
    const count = detail.sub_questions.length;
    addDone(`Planned the research (${detail.question_type}, ${count} sub-question${count > 1 ? "s" : ""})`);
    setPending("Choosing what to look up");
  } else if (node === "agent") {
    if (detail.requested_tools.length) {
      addDone(`Chose ${detail.requested_tools.join(", ")}`);
      setPending("Reading the labels");
    } else {
      addDone("Finished collecting evidence");
      setPending("Writing the answer");
    }
  } else if (node === "tools") {
    for (const call of detail.calls) {
      const outcome = call.error ? `failed: ${call.error}` : `${call.passages.length} passage(s)`;
      addDone(`${call.tool} (${shortArguments(call.arguments)}): ${outcome}`);
    }
    setPending("Deciding whether more evidence is needed");
  } else if (node === "write") {
    addDone(detail.revision ? "Rewrote the answer after review" : "Wrote a draft answer");
    setPending("Checking each claim against its source");
  } else if (node === "critique") {
    if (!detail.checked_by_llm) addDone("Found a citation that matches no retrieved passage");
    else addDone(detail.verdict === "pass" ? "Review approved every claim" : "Review asked for changes");
    setPending(detail.verdict === "pass" ? "Formatting the citations" : "Applying the review");
  }
}

// ---------------------------------------------------------- final result
function highlightSource(number) {
  for (const item of els.sourcesList.children) item.classList.toggle("is-highlighted", item.dataset.cite === number);
  for (const cite of els.answerBody.querySelectorAll(".cite")) {
    cite.setAttribute("aria-pressed", String(cite.dataset.cite === number));
  }
  const target = document.getElementById(`source-${number}`);
  if (target) target.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function renderSources(citations) {
  els.sourcesList.replaceChildren();
  for (const citation of citations) {
    const item = el("li", undefined, "source");
    item.id = `source-${citation.id}`;
    item.dataset.cite = String(citation.id);
    const doc = el("p", undefined, "source__doc");
    doc.append(el("span", `[${citation.id}]`), documentLabel(citation.file));
    item.append(doc, el("p", citation.section, "source__section"), el("p", citation.excerpt, "source__excerpt"));
    els.sourcesList.append(item);
  }
  els.sources.hidden = citations.length === 0;
}

function renderTrace(result) {
  const body = els.traceBody;
  body.replaceChildren();

  if (result.plan) {
    body.append(el("h4", `Plan: ${result.plan.question_type} question`));
    const steps = el("ol");
    for (const question of result.plan.sub_questions) steps.append(el("li", question));
    body.append(steps);
  }

  body.append(el("h4", `Tools chosen by the agent (${result.tool_calls.length})`));
  const table = el("table");
  const head = el("tr");
  for (const title of ["Tool", "Arguments", "Passages", "Time"]) head.append(el("th", title));
  table.append(head);
  for (const call of result.tool_calls) {
    const row = el("tr");
    const args = el("td");
    args.append(el("code", JSON.stringify(call.arguments)));
    row.append(
      el("td", call.tool),
      args,
      el("td", call.error ? call.error : String(call.passages.length), call.error ? "is-error" : undefined),
      el("td", `${call.duration_ms} ms`),
    );
    table.append(row);
  }
  body.append(table);

  const meta = result.meta;
  body.append(
    el("h4", "Run"),
    el(
      "p",
      `Model ${meta.model}. ${meta.llm_calls} language-model calls, ${meta.agent_steps} research steps, ` +
        `${meta.revisions} revision(s), ${meta.evidence_passages} passages read, ${(meta.latency_ms / 1000).toFixed(1)} s.`,
    ),
  );
}

function renderResult(result) {
  if (pending) pending.remove();
  pending = null;
  addDone(`Done in ${(result.meta.latency_ms / 1000).toFixed(1)} s`);

  els.verdict.dataset.verdict = result.review.verdict;
  els.verdict.textContent = VERDICTS[result.review.verdict];
  els.answerBody.innerHTML = renderMarkdown(result.answer);

  els.flagsList.replaceChildren(...result.review.flags.map((flag) => el("li", flag)));
  els.flags.hidden = result.review.flags.length === 0;
  els.disclaimer.textContent = result.disclaimer;

  renderSources(result.citations);
  renderTrace(result);
  els.answer.hidden = false;
}

function showError(message) {
  if (pending) pending.remove();
  pending = null;
  els.error.textContent = message;
  els.error.hidden = false;
}

// ---------------------------------------------------------------- requests
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = chunk.split("\n").find((line) => line.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)));
    }
  }
}

async function ask(question) {
  els.examples.open = false; // la réponse remonte juste sous le formulaire
  els.result.hidden = false;
  els.answer.hidden = true;
  els.error.hidden = true;
  els.sources.hidden = true;
  els.trace.open = false;
  els.progress.replaceChildren();
  pending = null;
  setPending("Planning the research");
  els.button.disabled = true;

  try {
    const response = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
      throw new Error(detail || `The server answered ${response.status}.`);
    }
    let finished = false;
    await readEvents(response, (event) => {
      if (event.type === "step") onStep(event);
      if (event.type === "result") {
        finished = true;
        renderResult(event.result);
      }
      if (event.type === "error") {
        finished = true;
        showError(event.message);
      }
    });
    if (!finished) showError("The answer stopped before the end. Try again.");
  } catch (error) {
    showError(error.message || "The agent could not be reached. Check that the server is running.");
  } finally {
    els.button.disabled = false;
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(String(response.status));
    const health = await response.json();
    els.status.dataset.state = "ready";
    els.status.textContent =
      `Ready. ${health.drugs.length} drugs and ${health.passages} passages indexed, ` +
      `semantic search ${health.semantic_search ? "on" : "off (keyword search only)"}, model ${health.model}.`;
  } catch {
    els.status.dataset.state = "down";
    els.status.textContent = "The agent is not reachable. Start the server, then reload this page.";
  }
}

// ------------------------------------------------------------------ wiring
els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = els.question.value.trim();
  if (question.length >= 3) ask(question);
});

els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) els.form.requestSubmit();
});

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    els.question.value = button.textContent;
    els.form.requestSubmit();
  });
}

els.answerBody.addEventListener("click", (event) => {
  const cite = event.target.closest(".cite");
  if (cite) highlightSource(cite.dataset.cite);
});

loadStatus();
