---
share: true
---
# Smart Pointers

The notes reflect the topics in
[Chapter 15](https://doc.rust-lang.org/book/ch15-00-smart-pointers.html)

This is a wild chapter since we don't even know what is a dumb pointer yet.

## Box<T>

`Box` is `std::unique_ptr`, of course better. Or, think it another way, a raw
pointer that enforce move semantics.

Rust doc helpfully summarize for us the usecases of such a data structure:

> -   When you have a type whose size can’t be known at compile time and you
>     want to use a value of that type in a context that requires an exact size
>     (e.g. Vector or String)
> -   When you have a large amount of data and you want to transfer ownership
>     but ensure the data won’t be copied when you do so (I want to make sure
>     the data lives for at least as long as I live, and I would not want to
>     copy it.)
> -   When you want to own a value and you care only that it’s a type that
>     implements a particular trait rather than being of a specific type (Rust
>     specific)

A good example of why we need unique pointer would be singly linked list. First,
we don't want to copy the data (node). Second, we can't have a type that
contains data of itself type. It would have infinite size.

## Deref trait

This is basically overriding that `*`.

```rust
use std::ops::Deref;

impl<T> Deref for MyBox<T> {
    type Target = T; // (1)

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

```

1. Associated type. This looks just like class namespaced type. That
   `typedef T value_type;`

So that when you do `*y` afterwards, what's really going on is `*(y.deref())*`.

### Implicit Deref Coercions with Functions and Methods

So this means implicit conversion to save you from typing `*` a lot. `&A` can be
converted to `&B` implicitly if deref of `A` returns B. So `&MyBox<String>` can
be used in functions accepting `&str`. Very sweet.

Just like in C++ you gonna do a version with `const` and a version without
`const`, you gonna do that with Rust too with `DerefMut`.

## Drop trait

RAII in C++, context manager in Python.

```rust
impl Drop for CustomSmartPointer {
    fn drop(&mut self) {
        println!("Dropping CustomSmartPointer with data `{}`!", self.data);
    }
}
```

Interestingly it offers `std::mem::drop` so you can call drop early before out
of scope. You do stuff like `drop(c)`. This can be used on, for example, a
`std::scoped_lock` in C++. Lock get released as we want.

## Rc<T>, the shared pointer.

```rust
fn main() {
    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    println!("count after creating a = {}", Rc::strong_count(&a)); // (2)
    let b = Cons(3, Rc::clone(&a)); // (1)
    println!("count after creating b = {}", Rc::strong_count(&a));
    {
        let c = Cons(4, Rc::clone(&a));
        println!("count after creating c = {}", Rc::strong_count(&a));
    }
    println!("count after c goes out of scope = {}", Rc::strong_count(&a));
}

```

1. `Rc::clone`, or the `.clone()` need to be called instead of just passing it
   (it would be a move then).
2. Reference count can be accessed this way.

Note that as we can't have multiple mutable reference, these pointers are
immutable. Also the check happens at compile time, unlike expensive
`std::shared_ptr`.

## RefCell<T>, the ???

-   `Rc<T>` enables multiple owners of the same data; `Box<T>` and `RefCell<T>`
    have single owners.
-   `Box<T>` allows immutable or mutable borrows checked at compile time;
    `Rc<T>` allows only immutable borrows checked at compile time; `RefCell<T>`
    allows immutable or mutable borrows checked at runtime.
-   Because `RefCell<T>` allows mutable borrows checked at runtime, you can
    mutate the value inside the `RefCell<T>` even when the `RefCell<T>` is
    immutable.

It's a wrapper that allows you to do interior mutability, like in C++ the
pointer is const while the data is not. By doing so it cannot do compile time
check. Note it still enforce only one mutable writer.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct MockMessenger {
        sent_messages: RefCell<Vec<String>>,
    }

    impl MockMessenger {
        fn new() -> MockMessenger {
            MockMessenger {
                sent_messages: RefCell::new(vec![]),
            }
        }
    }

    impl Messenger for MockMessenger {
        fn send(&self, message: &str) {
            self.sent_messages.borrow_mut().push(String::from(message)); // (1)
        }
    }

    #[test]
    fn it_sends_an_over_75_percent_warning_message() {
        // --snip--
        let mock_messenger = MockMessenger::new();
        let mut limit_tracker = LimitTracker::new(&mock_messenger, 100);

        limit_tracker.set_value(80);

        assert_eq!(mock_messenger.sent_messages.borrow().len(), 1);
    }
}

```

1. Notice the `borrow_mut()` here.

We can combine it with `Rc` so we got a mutable shared value:

```rust
let value = Rc::new(RefCell::new(5));
```

## Weak<T>, the weak pointer

Since `Rc` use reference count, it suffers from reference cycle. We need garbage
collection to prevent this from happening.

So the weak pointer in Rust is `Weak`. It can be get from `Rc` via
`Rc::downgrade`. And it can be `upgrade` into `Option<Rc<T>>`.
