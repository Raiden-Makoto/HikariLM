// What several parts of the page change: the open chat, the open file and the
// folder it is in. Modules can read each other's variables but not set them,
// so these live on one shared object.
export const state = {
  activeRequest: null,
  allFiles: [],
  attached: [],
  chatId: null,
  chats: [],
  filesCut: false,
  filesStarted: false,
  folderName: "",
  lastContext: null,
  messages: [],
  openPath: "",
  openToken: 0,
  openVersion: "",
  savedText: "",
  saving: false,
  source: null,
};
