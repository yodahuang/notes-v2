{pkgs, ...}: {
  languages.javascript = {
    enable = true;
    # The project uses npm instead of pnpm
    npm = {
      enable = true;
      install.enable = true;
    };
  };

  packages = [
    pkgs.watchexec
  ];

  processes = {
    build.exec = ''
      watchexec \
        --watch content \
        --watch quartz.config.ts \
        --watch quartz.layout.ts \
        --debounce 3s \
        --on-busy-update queue \
        -- npx quartz build
    '';

    serve.exec = ''
      npx browser-sync start --config bs-config.cjs
    '';
  };
}
