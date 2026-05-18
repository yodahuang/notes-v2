module.exports = {
  server: {
    baseDir: "public",
    serveStaticOptions: {
      extensions: ["html"],
    },
  },
  port: 8080,
  files: "public/**/*",
  notify: false,
  open: false,
}
