## Some Introduction

First, what's Nix?
Nix is:
- a programming language
- a program manager
- related to NixOS, an operating system
- a new way to package and build things.
- ???
Very confusing. I know. Read my other blog for more intro!

Darwin is Mac. Why it's darwin? Search yourself.

So `nix-darwin` is basically a way to configure your Mac just like NixOS. 
## What can actually be configured?
- `systemd`-like services. On Mac land they are `launchd` services.
- **SOME** default settings. By some I mean a very small portion. Things it cannot do includes:
	- Setting wallpaper
	- Set what's in the dock
	- Set if you want magnify effect in dock or not
	- Add new input method
- So you may wonder, what can it configure? Show file extension in Finder, remap Caps to Ctrl, etc. You can tell the one that picks the features still have the "Linux mindset". It's like the UI is not an important part of the system.
- Install apps. Via `nixpkgs`, `homebrew` or AppStore. More on that later.
## On the apps
While `nix-darwin` does introduce an easy way to specify programs from all the sources, using `homebrew`, cast or not, or AppStore, still defeats half of the purpose of Nix. You don't really get the reproducible build, or immutable config. You just get a fancy wrapper around `brew` that can be configured via a file.

Sadly you can't escape that. Almost all the GUI app requires installing from `homebrew` or AppStore. That's because `nixpkgs` cannot distribute XCode libraries, and all Mac native apps are using these.

An extreme example is our dear GNU Emacs.  In `nixpkgs`, as of June 2023, there's no Emacs build that can create a usable desktop entry.  Here's the [Github issue, lots of discussion](https://github.com/NixOS/nixpkgs/issues/127902). I'm not sure what I should feel.

Also related is that the programs on Mac are just tier 2 or tier 3. Quite some developers do not want to be hold back by broken darwin build. See [this heated discussion](https://discourse.nixos.org/t/darwin-again/29331).
### My preference on program sources
1. Nix package. Reproducible and easy to config.
2. AppStore. AppStore apps have more limitation on what they are allowed and not allowed to do to your system. So safer. Also come with free data sync and update via the walled garden.
3. `homebrew`, mad land.
## Why can't we get good stuff
???