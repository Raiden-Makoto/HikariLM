// Starts the page. Each part only declares things when it loads; its start()
// adds its listeners and first steps. They run here in this order, once
// every part has loaded, so nothing is used before it exists.
import { state } from "./state.js";
import { messageBox, narrow, showEmpty, start as startCore } from "./core.js";
import { start as startMarkdown } from "./markdown.js";
import { loadModels, start as startModels } from "./models.js";
import { refreshChats, start as startChatList } from "./chat-list.js";
import { draftKey, lastChatKey, loadDraft, openChat } from "./chats.js";
import { start as startSend } from "./send.js";
import { applyBackground, applyTheme, setSidebar, start as startPage } from "./page.js";
import { start as startSettings } from "./settings.js";
import { sectionToOpen, setSection, start as startSections } from "./sections.js";
import { start as startFolderView } from "./folder-view.js";
import { start as startFileActions } from "./file-actions.js";
import { start as startLayout } from "./layout.js";
import { start as startShortcuts } from "./shortcuts.js";

startCore();
startMarkdown();
startModels();
startChatList();
startSend();
startPage();
startSettings();
startSections();
startFolderView();
startFileActions();
startLayout();
startShortcuts();

applyTheme(localStorage.getItem("lightusagi-theme") || "light");
applyBackground(localStorage.getItem("lightusagi-bg") || "");
setSidebar(narrow.matches ? false : localStorage.getItem("lightusagi-sidebar") !== "closed");
showEmpty();
messageBox.focus();
setSection(sectionToOpen(), "replace");
await loadModels();
await refreshChats();
const last = localStorage.getItem(lastChatKey());
if (last) await openChat(last);
if (!state.chatId) localStorage.removeItem(lastChatKey());
await loadDraft(draftKey());
