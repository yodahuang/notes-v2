---
Created: 2022-03-11T11:09
Link: https://m.youtube.com/watch?v=rHIkrotSwcc
tags:
  - Finished
---
Simple talk. Three types of cost:

- Compile time cost: e.g. introduction of arena based protobuf
- Runtime cost: e.g. unique pointer. It should be copy by value, but doing so actually introduce a temporary variable. Generated machine code quite long. Passing by `&&` has double redirect.
- Human cost: smaller functions leads to looking up the function back and forth.

I still think compiled language’s near-zero abstraction is pretty good though. Like class method (without vtable) is just accessing that address.