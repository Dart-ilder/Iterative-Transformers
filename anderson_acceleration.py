import torch


def anderson_acceleration(args, f: callable, num_steps: int=5, m: int=5, tol: float=1e-4):
      """
      Perform an Anderson Acceleration step.

      Args:
          x (torch.Tensor): The current iterate.
          f (callable): The fixed-point function.
          m (int): The depth of the Anderson Acceleration memory.
          tol (float): The tolerance for convergence.
          max_iter (int): The maximum number of iterations.

      Returns:
          torch.Tensor: The next iterate.
      """
      x = args[0]
      F = [x]  # List to store function evaluations
      X = [x]  # List to store iterates

      for k in range(1, num_steps + 1):
          out = f(*args)
          x_next = out[0]
          F.append(x_next - x)
          X.append(x_next)

          # If the residual is small enough, return
          if torch.norm(F[-1]) < tol:
              return x_next

          # Update x using Anderson Acceleration
          if k >= m:
              # Construct the coefficient matrix
              T = torch.stack(F[-m:], dim=1)
              G = T.T @ T
              c = torch.linalg.solve(G, torch.ones(m, device=x.device))
              c /= c.sum()  # Normalize the coefficients

              # Update x
              x = torch.stack(X[-m:], dim=1) @ c
          else:
              x = x_next
              
          args[0] = x

      return x
