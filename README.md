# numerical-final

My algorithm for resolving a neutron lethargy spectrum problem in elemental iron (Fe) medium. Through a series of assumptions, the integro-differential equation known as the neutron continuity equation may be reduced to an entirely integral form.

Free neutrons entering a medium will sometimes collide with the nucleii of atoms that make up that medium. Zero-collision, one-collision and two-collision groups can be separated and solved analytically, while three or more collisions must be computed numerically. `final3.py` represents the final algorithm for solving this three or more collision bucket, while `final1.py` and `final2.py` consist of tests and examples of numerical conditioning with regards to certain trends, such as oscillation.

To solve for three or more collisions requires the use of a method of numerically inverting a Laplace transform known as a Bromwich integral which, when algebraically resolved to a usuable form, contains an infinite series that osciallates about zero on the `u` axis. Because the problem is ill-conditioned, an acceleration of convergence is achieved by using Aitken's delta-squared process. This allows for much faster convergence of the resulting function.

A full write-up of the math is available in `.pdf` form.

Special thanks to Vasily Arzhanov who was my professor for this class (Numerical Methods in Nuclear Engineering).
