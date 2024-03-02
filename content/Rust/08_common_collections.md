---
share: true
---
# Common Collections

The notes reflect the topics in
[Chapter 8](https://doc.rust-lang.org/book/ch08-00-common-collections.html)

## Vector

Our old friend `std::vector` here. Nothing special, just some Rust syntax here.

```rust
fn main() {
    // Create a vector, just like C++ normal constructor & initilaizer list
    let v: Vec<i32> = Vec::new();
    let v = vec![1, 2, 3]; // (1)

    // push_back
    let mut v = Vec::new();
    v.push(5);
    v.push(6);

    // Read element
    let v = vec![1, 2, 3, 4, 5];

    let third: &i32 = &v[2];
    println!("The third element is {}", third);

    match v.get(2) {
        Some(third) => println!("The third element is {}", third),
        None => println!("There is no third element."),
    }

    // Iterate
    for i in &v {
        println!("{}", i);
    }
    for i in &mut v {
        *i += 50; // Note the * here
    }
}
```

1. Why does this need to be a macro? That's because there's no variadic template
   in Rust (and I don't think C++'s way of recursive template metaprogramming is
   beautiful). So to make a function take arbitary number of parameters, you
   need to use macro.

Note we can use `enum` to put in various types in vector, just like
`std::variant`. `enum` is a tagged union though, so its size is slightly bigger
than the largest data it holds (as it need to store which type is being stored).

## String

Normal string, but not integer indexable.

```rust
fn main() {
    // Creating a new string
    let mut s = String::new();
    let s = "initial contents".to_string();
    let s = String::from("initial contents");

    // Updating a string
    let mut s = String::from("foo");
    s.push_str("bar");
    s.push('l'); // (1)

    let s1 = String::from("Hello, ");
    let s2 = String::from("world!");
    let s3 = s1 + &s2; // note s1 has been moved here and can no longer be used (2)
    let s = format!("{}-{}-{}", s1, s2, s3);

    // No integer indexing (3)

    // Slicing strings
    let hello = "Здравствуйте";
    let s = &hello[0..4];

    // Iterate over
    for c in "नमस्ते".chars() {
        println!("{}", c);
    }
    for c in "नमस्ते".bytes() {
        println!("{}", c);
    }
}
```

1. Why we have both `push_str` and `push`? That's because there's no overload in
   Rust. What we have is trait. But trait can't save us in this case. String and
   char doesn't share any common traits. If we use `enum` here, there'll be
   performance issues.

2. The signature is like

    ```rust
     fn add(self, s: &str) -> String {
    ```

3. UTF-8 string is tricky. The UTF-8 characters can have different byte length.
   And that's the reason Rust's string indexing is not a O(n) operation.
    - Python: Sacrifice memory, so that all the characters have the same length.
      It also determine whether to use ASCII or UTF-8 under the hood
      automatically. See
      [here](https://rushter.com/blog/python-strings-and-memory/) for more
      details.
    - C++: Has some support, but indexing is on byte level.
    - Emacs Lisp, Julia and Go: Either doing "optimization", or act kinda like
      C++. See [here](https://nullprogram.com/blog/2019/05/29/) for more
      details.

## HashMap

```rust
fn main() {
    use std::collections::HashMap;

    let field_name = String::from("Favorite color");
    let field_value = String::from("Blue");

    // Insert
    let mut map = HashMap::new();
    map.insert(field_name, field_value);
    // field_name and field_value are invalid at this point, try using them and
    // see what compiler error you get!

    // Get
    let field_name = "Favorite color".to_string();
    let value = scores.get(&field_name);
    for (key, value) in &map{
        println!("{}: {}", key, value);
    }
}

```
