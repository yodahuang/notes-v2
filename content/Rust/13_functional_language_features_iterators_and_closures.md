---
share: true
---
# Functional Language Features: Iterators and Closures

The notes reflect the topics in
[Chapter 13](https://doc.rust-lang.org/book/ch13-00-functional-features.html)

## Closures: Anonymous Functions that Capture Their Environment

Closure is C++ lambda and is better than the Python one.

Comparison of function syntax and closure one, with more elements omitted on
each line:

```rust
fn  add_one_v1   (x: u32) -> u32 { x + 1 }
let add_one_v2 = |x: u32| -> u32 { x + 1 };
let add_one_v3 = |x|             { x + 1 };
let add_one_v4 = |x|               x + 1  ;
```

It reference captures everything, deciding `mut` or not automatically. If you
want it to take ownership of an value, you need to call `move` manually. More on
that in Chapter 16.

### What's the type of closure?

We all know C++ lamda has a strange type. (It's not `std::function`, of course.
It's slow because type erasion.) What's this closure thing? Luckily we only care
about trait. Let's see what are the options:

-   `FnOnce` applies to closures that can be called at least once. All closures
    implement at least this trait, because all closures can be called. A closure
    that moves captured values out of its body will only implement FnOnce and
    none of the other Fn traits, because it can only be called once.
-   `FnMut` applies to closures that don’t move captured values out of their
    body, but that might mutate the captured values. These closures can be
    called more than once.
-   `Fn` applies to closures that don’t move captured values out of their body
    and that don’t mutate captured values, as well as closures that capture
    nothing from their environment. These closures can be called more than once
    without mutating their environment, which is important in cases such as
    calling a closure multiple times concurrently.

We use them like this:

```rust
impl<T> Option<T> {
    pub fn unwrap_or_else<F>(self, f: F) -> T
    where
        F: FnOnce() -> T
    {
        match self {
            Some(x) => x,
            None => f(),
        }
    }
}
```

## Processing a Series of Items with Iterators

Iterators are like Python ones + Java stream / C++ range. Not to be confused
with C++ iterator.

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn iterator_demonstration() {
        let v1 = vec![1, 2, 3];

        let mut v1_iter = v1.iter();

        assert_eq!(v1_iter.next(), Some(&1)); // (1)
        assert_eq!(v1_iter.next(), Some(&2));
        assert_eq!(v1_iter.next(), Some(&3));
        assert_eq!(v1_iter.next(), None);
    }
}
```

1. You might wonder: what does the `&1` mean here? Pointer to the literal `1`?
   Note this is pattern matching working here. What it means is "a `Some` emum,
   holding a reference to value 1".

### Consuming adapters

This means the adaper would advancing iterator till end.

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn iterator_sum() {
        let v1 = vec![1, 2, 3];

        let v1_iter = v1.iter();

        let total: i32 = v1_iter.sum();

        assert_eq!(total, 6);
    }
}
```

`v1_iter` cannot be accessed later because the `sum` method takes ownership.

### Iterator adapters

This is like Java `stream()` or converting C++ vector to a `range` and later
`transform` on it. They produce different iterators by changing some aspect of
the original iterator.

```rust
fn main() {
    let v1: Vec<i32> = vec![1, 2, 3];

    let v2: Vec<_> = v1.iter().map(|x| x + 1).collect();

    assert_eq!(v2, vec![2, 3, 4]);
}

```

### Performance

It's as fast as loop. That's expected as it's also the case for C++ and Python.
