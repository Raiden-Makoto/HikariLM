// The page on making a LoRA (/setup/lora): a Copy button on every command and on train.json.

import { copyButton } from "./copy.js";

for (const shown of document.querySelectorAll(".setup-command code, .setup-file code")) {
  const button = copyButton(() => shown.textContent.trim(), shown);
  const file = shown.closest(".setup-file");
  if (file) file.append(button);
  else shown.after(button);
}
