// A Copy button for a command or a file shown on a setup page.

export function copyButton(text, shown) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Copy";
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text());
      button.textContent = "Copied";
    } catch (error) {
      window.getSelection().selectAllChildren(shown);
      button.textContent = "Press Cmd+C";
    }
    setTimeout(() => { button.textContent = "Copy"; }, 2000);
  });
  return button;
}
