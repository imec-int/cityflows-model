from http.server import HTTPServer, BaseHTTPRequestHandler
import logging


class CustomHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()


def start(port=80, server_class=HTTPServer, handler_class=CustomHandler):
    logger = logging.getLogger("root")
    try:
        server_address = ('', port)
        logger.info('Server up and running on port: %s' % port)
        httpd = server_class(server_address, handler_class)
        httpd.serve_forever()
    except Exception as error:
        logger.error(error)
        exit()
