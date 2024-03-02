---
share: true
---
# Understand Ownership

The notes reflect the topics in
[Chapter 4](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)

This is one of the core concept of Rust the programming language (if not the
most important one).

To truly understand how it works, we need to understand "why it is designed to
work like this". There's some common operations every language need to express.
We'll use C++, Rust and Python here to see how each language deal with it.

## What does assignment mean?

=== "Rust"

    In Rust, assignment means move, unless the object implements `Copy` trait. The
    prerequisite of that is the object satisfy conditions similar to
    `std::is_trivially_destructible`. Moved variable is guaranteed to not be used
    via compile time checking (why doesn't C++ have this?).

=== "C++"

    Assignment means copy (unless it's moved).
    [Copy ellision](https://en.cppreference.com/w/cpp/language/copy_elision) may
    kick in to avoid `const int i = 3` from being expensive, but still, the idea is
    that you copy whatever is stored in the object. And then we got this (non RAII
    style object):

    ```cpp
    class Foo {
        Foo(Bar * bar) {...}
        ~Foo {
            delete bar;
        }
    }

    Foo foo_1;
    Foo foo_2(foo_1);

    // And then we got double memory deallocation.
    ```

=== "Python"

    In Python, assignment means creating a mutable aliase of a value (in CPython,
    all the values are stored in a internal heap). Assignment never copies data.
    Changing a value may be interpreted as rebinding an alias to the new value. All
    the values are managed in a `shared_ptr` style (just speaking of CPython). Thus
    we get the seemingly surprising behavior:

    ```python
    for x in array:
        x+=2  # Nothing really happens
    ```

    That's because there's no real "references" in Python. All the things we are
    playing around is just names and values. Names can never refer to other names.
    It can only refer to value.

    For a more detailed explanation, see [here](https://nedbatchelder.com/text/names1.html).

## How to deal with iterator invalidation?

=== "Rust"

    The problem is easy to deal with if it simply does not exist.

    ```rust
    fn main() {
    let mut buf = vec![1,2,3,4];

    for i in &buf {
        buf.push(*i);
    }
    }
    ```

    gives us

    ```
    error[E0502]: cannot borrow `buf` as mutable because it is also borrowed as immutable
    --> main.rs:5:7
      |
    4 |   for i in &buf {
      |            ----
      |            |
      |            immutable borrow occurs here
      |            immutable borrow later used here
    5 |       buf.push(*i);
      |       ^^^^^^^^^^^^ mutable borrow occurs here
    ```

    Ok. What if we say `for i in &mut buf`?

    ```
    error[E0499]: cannot borrow `buf` as mutable more than once at a time
    --> main.rs:5:7
      |
    4 |   for i in &mut buf {
      |            --------
      |            |
      |            first mutable borrow occurs here
      |            first borrow later used here
    5 |       buf.push(*i);
      |       ^^^ second mutable borrow occurs here
    ```

=== "C++"

    You deal with everything yourself :crying_cat_face:.

    ```cpp
    vector<int> x = { 1, 2, 3 };
    for (auto it = x.begin(); it != x.end(); ++it) {
        x.push_back(42);
    }
    // Vector resize and then the iterator points to nowhere.
    ```

=== "Python"

    Python doesn't really have the idea of iterator. With its for loop, it behave as
    we expected, but the official guide mentioned

    > Code that modifies a collection while iterating over that same collection can
    > be tricky to get right. Instead, it is usually more straight-forward to loop
    > over a copy of the collection or to create a new collection

    Here are examples of
    [appending element](https://stackoverflow.com/questions/3752618/python-adding-element-to-list-while-iterating)
    and
    [removing element](https://stackoverflow.com/questions/6260089/strange-result-when-removing-item-from-a-list-while-iterating-over-it)
    when iterating a list. TLDR: Python, or say, CPython, in the current state, can
    somehow advance its interval iterator without invalidate the iterator (maybe its iterator contains a reference to the list itself). But this
    behavior should not be relied on and is considered dangerous.

## How to deal with dangling pointer?

=== "Rust"

    ```rust
    fn main() {
        let mut buf = vec![1, 2, 3, 4];
        let reference = &buf[0];
        buf.clear();
        println!("Got {}", reference);
    }
    ```

    gives

    ```
    error[E0502]: cannot borrow `buf` as mutable because it is also borrowed as immutable
    ```

=== "C++"

    404

=== "Python"

    All the things in Python is basically `shared_ptr`. That means existing references would always be valid as long as there's something holds on to it.

For more discussion on Rust's choice to implement this read write lock (even in
single threaded case), see
[this post](https://manishearth.github.io/blog/2015/05/17/the-problem-with-shared-mutability/).

# The slice type

```rust
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}

fn main() {
    let my_string = String::from("hello world");

    // `first_word` works on slices of `String`s, whether partial or whole
    let word = first_word(&my_string[0..6]);
    let word = first_word(&my_string[..]);
    // `first_word` also works on references to `String`s, which are equivalent
    // to whole slices of `String`s
    let word = first_word(&my_string);

    let my_string_literal = "hello world";

    // `first_word` works on slices of string literals, whether partial or whole
    let word = first_word(&my_string_literal[0..6]);
    let word = first_word(&my_string_literal[..]);

    // Because string literals *are* string slices already,
    // this works too, without the slice syntax!
    let word = first_word(my_string_literal);
}

```

There're other slices too:

```rust
let a = [1, 2, 3, 4, 5];
let slice = &a[1..3]; // Of type &[i32]
```

So the slices are just like `std::span` (for string it's more like
`std::string_view`), but with Python-like syntax sugar. An interesting thing is
string literals are string slices too. Their owner is a
[pre-allocated read-only memory](https://blog.thoughtram.io/string-vs-str-in-rust/).
No special `char[]`. Just simple string slice (which is a pointer and a length).
