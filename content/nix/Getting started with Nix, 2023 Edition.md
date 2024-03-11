
We need flake-first, up-to-date and deep tutorials.

## Zero to Nix

First, start with [Zero to Nix](https://zero-to-nix.com/). It's a new guide provided by Determinate System. When it published, there's quite [some controversies](https://discourse.nixos.org/t/parting-from-the-documentation-team/24900) around it.
The doc would teach you "how do I use a pre-made nix-flake", but not "how do I make one". It's basically a glorified cli `--help`.

## nix.dev

This one is the official new doc site, and it covers stuff in much finer details. As of the time of writing it's still not using flakes yet. But the gems are in the [language guide](https://nix.dev/tutorials/first-steps/nix-language). The language does not have a formal spec, just a lexer as guide. (You'll be surprised how many languages does not have them. E.g. Python)
The Nix language is a functional one, like Haskell. Guix on the other hand uses Guile, which may be better?

## Others
- [A detailed en / zh_cn book](https://nixos-and-flakes.thiscute.world) covering almost all the stuff. The author really spend a lot of time updating their blog in dual language.
- Xe's [blog](https://xeiaso.net/blog/nix-flakes-1-2022-02-21). Xe is an interesting guide and their blog is always a fun read.
- A [blog post](https://xyno.space/post/nix-darwin-introduction) focusing on nix-darwin

## Or maybe a higher level abstraction is good enough?

If you only need `nix shell` ish experience on a project level, [devbox](https://www.jetpack.io/devbox/) or [devenv](https://devenv.sh/) is a perfectly good solution. You don't need to write the flakes yourself with a cleaner interface.