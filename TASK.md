Task 5. PINN (Physics Informed Neural Network) for the Silverbox
oscillator
The Silverbox is a laboratory electronic circuit built to act as a Duffing oscillator: a second-order
system with a spring that stiffens as the displacement grows. The same equation shows up for a
mass on a cubic spring, and the circuit was built so that the equation can be measured cleanly. The
public record is the Silverbox benchmark of Wigren and Schoukens (European Control Conference,
4
2013), file SNLS80mV.mat, available from https://www.nonlinearbenchmark.org/. Channel V1 is
the input voltage u(t). Channel V2 is the output voltage y(t). Both are sampled at
fs = 610.35 Hz, ∆t = 1/fs.
The file contains 131,072 samples. The estimation experiment is a random-phase multisine, and it
begins at sample 40650 (indices into the file, starting at 0).
Problem.
The excerpt. Use the 4096 samples starting at index 42650, that is the slice [42650 : 46746].
This sits 2000 samples after the multisine begins. Subtract the mean of u on this excerpt and the
mean of y on this excerpt, and use those centered signals afterwards. The first 3072 samples are the
training record, about 5.03 s. The last 1024 samples are the test record, about 1.68 s. On the test
record the input u is known, because it is the voltage that was applied. The output y is what the
model has to produce.
The differential equation. Write the centered output as y and the centered input as u. The circuit
is modeled by
y¨ + a y˙ + b y + c y3 = g u(t). (1)
The four coefficients are properties of the circuit.
• a multiplies the velocity. It is the damping. On this circuit it is a few tens of s
−1
.
• b multiplies y. It is the linear stiffness, in s
−2
. The output of this excerpt is dominated by
frequencies near 70 Hz.
• c multiplies y
3
. It is the cubic stiffness, the term a linear spring does not have. At the peaks
of this excerpt, where |y| is near 0.2 V, the cubic force is a visible fraction of the linear force.
• g multiplies the input voltage. It is an input gain, because u and y are voltages on opposite
sides of the circuit.
They are not the weights of the network. The acceleration is produced by a network that has to
represent this force.
The network. The acceleration is a multilayer fully connected network, implemented in PyTorch.
You choose the depth, the widths, the activation, and how the inputs and the output are scaled.
With velocity v = y˙, the state is the pair (y, v) and
y˙ = v, v˙ = αθ(y, v, u). (2)
Integrate that state at step ∆t, driven by the measured input. Classical Runge–Kutta of order 4
is a sound choice for the step. The excerpt lasts 6.7 s near 70 Hz, several hundred cycles, so the
integrator is what draws the waveform. Try more than one architecture, and for each architecture
tune the training hyperparameters: the optimizer, the learning rate, the number of steps, and the
length of the rollout you train on. Keep the version with the best test score under the metric below.
The metric. After the weights of an architecture are fixed, do one free run of the whole excerpt.
Start at the first sample, with the measured position and a finite-difference velocity, and integrate
through the training samples and on into the test samples. Keep the same state when the test segment
begins. On the test segment the rollout may read u, and it may not read y. The performance of the
architecture is the root-mean-square error of this free run on the 1024 test samples,
RMSE =
vuut
1
1024
1024
X
i=1

yˆi − yi
2
.
5
A smaller test RMSE is a better version. Predicting zero scores the RMS of the test output, about
0.06 V, so that number is the baseline. Also compute the same free-run RMSE on the 3072 training
samples. A network that fits the training record and then drifts on the test segment shows up as a
large gap between the two.
In the solution PDF, describe the architecture you kept and the hyperparameters you settled on, the
state equation, and how you trained it. Say which other settings you tried and what test RMSE each
one got. Give both free-run RMSE numbers for the kept version, and a plot of measured against
predicted output on the test segment. The prediction in that plot is the free run above. Push the
code to a GitHub repository and put the link in the PDF. 12 points