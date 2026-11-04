import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from helper import imagegen, server


class BugReachesThePageTest(unittest.TestCase):
    """A bug in the helper used to close the connection, and the page said only "Failed to fetch"."""

    def setUp(self):
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def test_bug_before_the_reply_is_a_json_error(self):
        with mock.patch.object(imagegen, "clean_request", side_effect=RuntimeError("boom")), \
                mock.patch("traceback.print_exc"):
            request = urllib.request.Request(self.base + "/api/generate", b'{"model": "x", "prompt": "a fox"}',
                                             {"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 500)
        body = json.loads(caught.exception.read())
        self.assertIn("The helper hit a bug: RuntimeError: boom", body["error"])


if __name__ == "__main__":
    unittest.main()
