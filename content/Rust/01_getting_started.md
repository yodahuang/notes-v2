---
share: true
---
# Getting Started

The notes reflect the topics in
[Chapter 1](https://doc.rust-lang.org/book/ch01-00-getting-started.html)

This is a rather easy chapter. But still we can see Rust just learns from
languages from previous generation and improves on that.

### rustup

rustup seem to take its inspiration from [RVM](https://rvm.io). The ruby version
manager has since inspired NVM & N in Node.js world, pyenv in Python world. (Is
there even such a thing in C++ land?). The great thing about it is its' the
first language that I know of is implemented by first party.

### cargo

Cargo is inspired by [npm](https://www.npmjs.com) (I think). npm was the first
package manager that I know of that introduced per-project dependency management
(and brings along [memes](https://i.redd.it/tfugj4n3l6ez.png) about its size).
On Python land we got [PEP 582](https://www.python.org/dev/peps/pep-0582/) and
an implementation of that, [PDM](https://github.com/pdm-project/pdm).
[poetry](https://python-poetry.org) is not exactly like that.

#### TOML

Both cargo and poetry uses [TOML](https://toml.io/en/) as its configuration
format, which is sensible and better than npm's JSON. What's the best solution
for configuration has long been a heated topic on HackerNews, there's JSON,
YAML, TOML, or other script based solution. The downside of TOML is that it is
not very readable if we have a complex, multi-layer configuration, like
Kubernetes declarations. This is not the case with simple package manager
though.

#### where's cargo add?

This is the [#4 issue](https://github.com/rust-lang/cargo/issues/4), and then
tracked in [#5586](https://github.com/rust-lang/cargo/issues/5586). The reason
it has not been implemented is because those devs working on cargo tries to make
it absolutely performant and best. A big obstacle is to maintain the comments in
the original config. A normal TOML parser would just ignore the comment, which
is not ideal (`peotry` seems fine with that though).

BTW, I don't think there's any YAML parser that can read the document and write
it back without changing a bit. There's just too many ways to do a thing.

#### How does cargo avoid excessive dependencies, or huge unnecessary dependencies

TBD
