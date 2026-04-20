const http = require('http');

const server = http.createServer((req, res) => {
  res.end('Hello from Docker 🚀');
});

server.listen(5005, () => {
  console.log('Server running on port 5005');
});