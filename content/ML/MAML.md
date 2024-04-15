---
share: true
---
> - Creates a copy of the initialization weights
> - Runs an iteration of gradient descent for a random task on the copy
> - Backpropagates the loss on a _test_ set through the **iteration of** gradient descent and back to the initial weights, so that we can update the initial weights in a direction in which they would have been easier to update.

So basically it's back propagagion throuth time, or [[BPTT]], across different tasks. 
```python
def maml_sine(model, epochs, lr_inner=0.01, batch_size=1):
    optimizer = torch.optim.Adam(model.params())
    
    for _ in range(epochs):
        for i, t in enumerate(random.sample(SINE_TRAIN, len(SINE_TRAIN))):
            new_model = SineModel()
            new_model.copy(model, same_var=True)
            loss = sine_fit1(new_model, t, create_graph=not first_order)
            for name, param in new_model.named_params():
                grad = param.grad
                new_model.set_param(name, param - lr_inner * grad)
                        
            sine_fit1(new_model, t, force_new=True)

            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
```
