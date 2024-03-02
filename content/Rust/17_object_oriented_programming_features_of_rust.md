---
share: true
---
# Object-Oriented Programming Features of Rust

The notes reflect the topics in
[Chapter 17](https://doc.rust-lang.org/book/ch17-00-oop.html)

## Using Trait for dynamic dispatch

Let's say we want to have a gui lib, which support drawing a bunch of
components. How would we implement that?

Let's consider C++ first. An good example would be the
[Qt framework](https://www.qt.io/) (let's forget Qt QML for now). Stuff there
are all `Widget`, and stuff got complicated inheritence chain. Here's an
[example](https://github.com/qt/qtbase/blob/516871d3e5d7815c0fb4a93c71f8cb35f7b3325c/src/widgets/widgets/qtoolbutton.h).

Recall that we need dynamic dispatch so that we can do stuff like

```c++
// C inherits B

B* b = new C();
b->bar(); // call C's method if bar is a virtual function.
```

Virtual in C++ basically means dymamic dispatch. The compiler stores a vpointer
in each class, add `sizeof(vpointer)` to each class. The pointer points to
`vtable`, which maps function calls to the right function pointer address.

What does Rust do? We don't have inheritence here. All we got is trait. So we
have dynamic dispatch that look like duck typing. Still using vtable
[under the hood](https://doc.rust-lang.org/1.30.0/book/first-edition/trait-objects.html#representation)
though.

```rust
pub trait Draw {
    fn draw(&self);
}

pub struct Screen {
    pub components: Vec<Box<dyn Draw>>, // (1)
}

```

1. That `dyn` means
   [dynamic dispatch](https://doc.rust-lang.org/std/keyword.dyn.html). It's
   needed in order to make a trait "type-like". This is similar to the case when
   we use traints in a method bound, we do `item: &impl Summary`. Of course
   `Box` is needed to make it a pointer.

What's the difference vs using generic type (template)? With template like

```rust
pub trait Draw {
    fn draw(&self);
}

pub struct Screen<T: Draw> {
    pub components: Vec<T>,
}

impl<T> Screen<T>
where
    T: Draw,
{
    pub fn run(&self) {
        for component in self.components.iter() {
            component.draw();
        }
    }
}
```

Only one typeo of the component would be initiated.

## Implementing an Object-Oriented Design Pattern

This part of the book is quite cryptic. Pay close attention to my annotation of
the code to understand some design choice over others.

The example given is to write a blog post struct that does this thing:

```rust
use blog::Post;

fn main() {
    let mut post = Post::new();

    post.add_text("I ate a salad for lunch today");
    assert_eq!("", post.content());

    post.request_review();
    assert_eq!("", post.content());

    post.approve();
    assert_eq!("I ate a salad for lunch today", post.content());
}
```

Let's first have a special state trait to do all the state transition stuff. It
looks like this:

```rust
pub struct Post {
    state: Option<Box<dyn State>>, // (1)
    content: String,
}

impl Post {
    pub fn new() -> Post {
        Post {
            state: Some(Box::new(Draft {})),
            content: String::new(),
        }
    }

    pub fn add_text(&mut self, text: &str) {
        self.content.push_str(text);
    }

    pub fn content(&self) -> &str {
        self.state.as_ref().unwrap().content(self) // (3)
    }

    pub fn request_review(&mut self) {
        if let Some(s) = self.state.take() { // (4)
            self.state = Some(s.request_review())
        }
    }

    pub fn approve(&mut self) {
        if let Some(s) = self.state.take() {
            self.state = Some(s.approve())
        }
    }
}

trait State {
    fn request_review(self: Box<Self>) -> Box<dyn State>; // (2)
    fn approve(self: Box<Self>) -> Box<dyn State>;

    fn content<'a>(&self, post: &'a Post) -> &'a str {
        ""
    }
}


struct Draft {}

impl State for Draft {
    fn request_review(self: Box<Self>) -> Box<dyn State> {
        Box::new(PendingReview {})
    }

    fn approve(self: Box<Self>) -> Box<dyn State> {
        self
    }
}

struct PendingReview {}

impl State for PendingReview {
    fn request_review(self: Box<Self>) -> Box<dyn State> {
        self
    }

    fn approve(self: Box<Self>) -> Box<dyn State> {
        Box::new(Published {})
    }
}

struct Published {}

impl State for Published {
    fn request_review(self: Box<Self>) -> Box<dyn State> {
        self
    }

    fn approve(self: Box<Self>) -> Box<dyn State> {
        self
    }

    fn content<'a>(&self, post: &'a Post) -> &'a str {
        &post.content
    }
}

```

1. Why the `Option` here? Please look at the annotation at `request_review`
   method.

2. Why no default method is being pr
3. ovided? If I provide one, the compiler would complain:

    ```
     error[E0277]: the size for values of type `Self` cannot be known at compilation time
     --> main.rs:38:9
     |
     37 |     fn request_review(self: Box<Self>) -> Box<dyn State> {
     |                                                         - help: consider further restricting `Self`: `where Self: std::marker::Sized`
     38 |         self
     |         ^^^^ doesn't have a size known at compile-time
     |
     = help: the trait `std::marker::Sized` is not implemented for `Self`
     = note: to learn more, visit <https://doc.rust-lang.org/book/ch19-04-advanced-types.html#dynamically-sized-types-and-the-sized-trait>
     = note: required for the cast to the object type `dyn State`
    ```

    So even though `Box` is actually sized, Rust does not allow us to return a
    Box of unknown size in the trait. It's related to somenthing called object
    safety.

    > The idea of "object-safety" is that you must be able to call methods of
    > the trait the same way for any instance of the trait. So the properties
    > guarantee that, no matter what, the size and shape of the arguments and
    > return value only depend on the bare trait — not on the instance (on Self)
    > or any type arguments (which will have been "forgotten" by runtime). (From
    > [Reddit thread](https://www.reddit.com/r/rust/comments/kw8p5v/what_does_it_mean_to_be_an_objectsafe_trait/))

4. We call the `as_ref` method on the `Option` because we want a reference to
   the value inside the `Option` rather than ownership of the value. Because
   state is an `Option<Box<dyn State>>`, when we call `as_ref`, an
   `Option<&Box<dyn State>>` is returned. If we didn’t call `as_ref`, we would
   get an error because we can’t move `state `out of the borrowed `&self` of the
   function parameter.

    We then call the `unwrap` method, which we know will never panic, because we
    know the methods on `Post` ensure that `state` will always contain a `Some`
    value when those methods are done.

5. > To consume the old state, the `request_review` method needs to take
   > ownership of the `state` value. This is where the `Option` in the `state`
   > field of `Post` comes in: we call the `take` method to take the `Some`
   > value out of the `state` field and leave a `None` in its place, because
   > Rust doesn’t let us have unpopulated fields in structs. This lets us move
   > the `state` value out of `Post` rather than borrowing it. Then we’ll set
   > the post’s `state` value to the result of this operation.

    > We need to set `state` to `None` temporarily rather than setting it
    > directly with code like `self.state = self.state.request_review();` to get
    > ownership of the `state` value. This ensures `Post` can’t use the old
    > `state` value after we’ve transformed it into a new state.

    There's
    [this StackOverFlow question](https://stackoverflow.com/questions/57193489/why-do-we-need-to-call-take-for-optiont-variable)
    about what does this actually mean. I don't find that useful. My guess is
    that if panic happen in the middle, there could be a state where the state
    is being moved out, but no new value is being assigned. Then we set
    `self.state` to this undefined value, which is again undefined. That's why
    we need to explicitly set a `None` there.

Instead of doing things like this, we can also just forget the state and just
have method on individual state struct (isn't that more intuitive)?

```rust

pub struct Post {
    content: String,
}

pub struct DraftPost {
    content: String,
}

impl Post {
    pub fn new() -> DraftPost {
        DraftPost {
            content: String::new(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

impl DraftPost {
    pub fn add_text(&mut self, text: &str) {
        self.content.push_str(text);
    }

    pub fn request_review(self) -> PendingReviewPost {
        PendingReviewPost {
            content: self.content,
        }
    }
}

pub struct PendingReviewPost {
    content: String,
}

impl PendingReviewPost {
    pub fn approve(self) -> Post {
        Post {
            content: self.content,
        }
    }
}

```
