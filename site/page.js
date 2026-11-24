const configured = window.projectLinks?.codeRepository;
if (configured) {
  try {
    const url = new URL(configured);
    if (url.protocol === "https:" && url.hostname === "github.com") {
      for (const link of document.querySelectorAll("[data-code], [data-code-row]")) {
        link.href = url.href;
        link.removeAttribute("download");
      }
      document.querySelector("[data-code]").textContent = "Code on GitHub ↗";
      document.querySelector("[data-code-label]").textContent = "GitHub repository ↗";
    }
  } catch { /* Retain the downloadable code archive for an invalid URL. */ }
}

const tasks = JSON.parse(document.querySelector("#task-values").textContent);
const picker = document.querySelector("#task-picker");
const explorer = document.querySelector("#task-explorer");
for (const [index, task] of tasks.entries()) {
  picker.add(new Option(task.suite + " · " + task.instruction, String(index)));
}
const selected = tasks.findIndex(task => task.suite === "Long" && task.id === 8);
picker.selectedIndex = Math.max(0, selected);
function showTask() {
  const task = tasks[picker.selectedIndex];
  document.querySelector("#current-instruction").textContent = task.instruction;
  document.querySelector("#task-frozen").textContent = task["frozen-base"] + " → " + task["frozen-finetuned"];
  document.querySelector("#task-joint").textContent = task["full-base"] + " → " + task["full-finetuned"];
}
showTask();
picker.addEventListener("change", showTask);
explorer.hidden = false;
