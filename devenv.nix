{pkgs, ...}: {
  languages.javascript = {
    enable = true;
    # The project uses npm instead of pnpm
    npm = {
      enable = true;
      install.enable = true;
    };
  };
}
