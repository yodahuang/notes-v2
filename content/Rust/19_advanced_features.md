---
share: true
---
# Advanced Features

The notes reflect the topics in
[Chapter 19](https://doc.rust-lang.org/book/ch19-00-advanced-features.html)

## Unsafe Rust

Unsafe rust still have borrow checker or other safety checks, but it enables the
following ability:

-   Dereference a raw pointer
-   Call an unsafe function or method
-   Access or modify a mutable static variable
-   Implement an unsafe trait
-   Access fields of unions

We can wrap unsafe code in a safe API. Here the "safe" mean nothing can ever be
wrong no matter the input. It's entirely determined by the author though. So I
can write a totally unsafe code ant declare "this is safe". So evil.

### Dereferencing a raw pointer

```rust
let mut num = 5;

let r1 = &num as *const i32;
let r2 = &mut num as *mut i32;

let address = 0x012345usize;
let r = address as *const i32;

unsafe {
    println!("r1 is: {}", *r1);
    println!("r2 is: {}", *r2);
}
```

### Calling an Unsafe Function or Method

Using the `split_at_mut` function in standard library as an example: it takes
one slice and makes it two by splitting the slice at the index given as an
argument. Its usage is like this:

```rust
fn main() {
    let mut v = vec![1, 2, 3, 4, 5, 6];

    let r = &mut v[..];

    let (a, b) = r.split_at_mut(3);

    assert_eq!(a, &mut [1, 2, 3]);
    assert_eq!(b, &mut [4, 5, 6]);
}
```

How do we implement that? It need to be unsafe because we cannot have two
mutable reference on the same object. So it's implemented like this:

```rust
use std::slice;

fn split_at_mut(values: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();

    assert!(mid <= len);

    unsafe {
        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

```

Note the function itself is not `unsafe`. That's because we as the developer
know the `ptr` and `len` must be good and this code is actually always safe. In
comparison, `from_raw_parts_mut` is unsafe because there's no guarantee that the
pointer or the length would be a good one.

### Extern

That's related to the built in FFI.

```rust
extern "C" {
    fn abs(input: i32) -> i32;
}

fn main() {
    unsafe {
        println!("Absolute value of -3 according to C: {}", abs(-3));
    }
}
```

??? note "Calling Rust from other languages"

    Do stuff like this:
    ```rust
    #[no_mangle]
    pub extern "C" fn call_from_c() {
        println!("Just called a Rust function from C!");
    }
    ```
    and then compile it to a shared library and link it.

### Accessing or modifying a mutable static variable

Recall that we can have `const` in global namespace (See
[Chapter 3](03_common_programming_concepts.md#variables-and-mutability)).

What if we want to have a mutable global variable?

```rust
static mut COUNTER: u32 = 0;

fn add_to_count(inc: u32) {
    unsafe {
        COUNTER += inc;
    }
}

fn main() {
    add_to_count(3);

    unsafe {
        println!("COUNTER: {}", COUNTER);
    }
}
```

### Implementing unsafe traint

```rust
unsafe trait Foo {
    // methods go here
}

unsafe impl Foo for i32 {
    // method implementations go here
}

fn main() {}
```

## Advanced traits

### Associated types

Associated type are not like `map::value_type`. They are a new thing in Rust
that are just placeholder, kinda like generic but more... restricted?

```rust
struct Counter {
    count: u32,
}

impl Counter {
    fn new() -> Counter {
        Counter { count: 0 }
    }
}

impl Iterator for Counter {
    type Item = u32; // (1)

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < 5 {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}
```

1. The associated type.

Why not

```rust
pub trait Iterator<T> {
    fn next(&mut self) -> Option<T>;
}
```

?

That's because we don't want `impl Iterator<String> for Counter` etc. Only one
iterator should be implemented for the given type. In that sense, it's more
restricted, but good.

### Default type parameters (and operator overloading)

```rust
use std::ops::Add;
impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// What's the definition of Add BTW? It's this.
trait Add<Rhs=Self> { // (1)
    type Output;

    fn add(self, rhs: Rhs) -> Self::Output; // (2)
}
```

1. By default we assume the type is self, unless you do stuff like
   `impl Add<Meters> for Millimeters`. Why we have to do stuff like this? That's
   because there's
   [no function overloading](03_common_programming_concepts.md#common-programming-concepts).

2. Tell Rust the output type by setting the `Output`.

### Fully qualified syntax for calling methods with the same name

```rust
fn main() {
    let person = Human;
    Pilot::fly(&person); // (1)
    Wizard::fly(&person);
    person.fly();
    println!("A baby dog is called a {}", <Dog as Animal>::baby_name()); (2)
}
```

1. This is for "noraml" instance methods.
2. This is for associated method (static method). `<Dog as Animal>` tells that
   we want to use the implementation of `Animal` for `Dog` as opposed to the
   implementation of `Animal` for some other type.

### Supertraits

This is to allow traits "inherit" other traits. For example, in A trait we call
method defined in B trait. In C++ it is all wild. The complier would just check
if a method is there due to SFIANE. Not here in Rust.

```rust
use std::fmt;

trait OutlinePrint: fmt::Display {
    ...
}
```

### Newtype pattern

For implementing a trait on a data structure, we requires that the library own
either of them. If we don't have ownership over either of them, we can wrap it.

```rust
use std::fmt;

struct Wrapper(Vec<String>);

impl fmt::Display for Wrapper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}]", self.0.join(", "))
    }
}

fn main() {
    let w = Wrapper(vec![String::from("hello"), String::from("world")]);
    println!("w = {}", w);
}
```

If we still want to use vector method, either implement it ourselves (like in
C++), or implement the `Deref` trait.

## Advanced types

