# State-based travel-time routing cost

Dynamic and stochastic assignment use a link cost with the physical meaning of instantaneous travel time.

For cell \((\ell,i)\), the current density is

\[
\rho(\ell,i,t)=\frac{n(\ell,i,t)}{\Delta x(\ell,i)}.
\]

The local composition-dependent triangular fundamental diagram is

\[
q(\ell,i,\rho,t)=\min\left\{
v^{\mathrm{ff}}(\ell)\rho,
\;w(\ell,i,t)\left(\rho^{\mathrm{jam}}-\rho\right)
\right\},
\]

where

\[
w(\ell,i,t)=\frac{L_{\mathrm v}}{\bar\tau(\ell,i,t)}.
\]

The instantaneous cell speed is then

\[
u(\ell,i,t)=
\begin{cases}
v^{\mathrm{ff}}(\ell), & \rho(\ell,i,t)=0,\\[4pt]
\dfrac{q(\ell,i,\rho(\ell,i,t),t)}{\rho(\ell,i,t)},
& \rho(\ell,i,t)>0.
\end{cases}
\]

Equivalently, for a nonempty cell,

\[
u(\ell,i,t)=
\begin{cases}
v^{\mathrm{ff}}(\ell),
& 0<\rho(\ell,i,t)\le \rho^{\mathrm{cr}}(\ell,i,t),\\[6pt]
\dfrac{w(\ell,i,t)\left(\rho^{\mathrm{jam}}-\rho(\ell,i,t)\right)}{\rho(\ell,i,t)},
& \rho(\ell,i,t)>\rho^{\mathrm{cr}}(\ell,i,t).
\end{cases}
\]

For numerical robustness, MIXSim applies a small positive speed floor \(\varepsilon_u=0.5\,\mathrm{m/s}\):

\[
\tilde u(\ell,i,t)=\max\{u(\ell,i,t),\varepsilon_u\}.
\]

The state-based link cost is the sum of the cell travel times,

\[
\kappa(\ell,t)
=
\sum_{i=1}^{I_\ell}
\frac{\Delta x(\ell,i)}{\tilde u(\ell,i,t)}.
\]

Hence \(\kappa(\ell,t)\) is measured in seconds. It is an instantaneous snapshot estimate: it is the travel time that would result if the current cell states persisted while the link were traversed. It is not an experienced or predictive travel time.

For dynamic shortest-path assignment, the outgoing alternative with minimum remaining travel-time cost is selected. For stochastic assignment, the remaining cost of choosing outgoing link \(k\) is

\[
C(\ell,k,h,t)=\kappa(k,t)+d(\operatorname{head}(k),r(h),t),
\]

where \(d\) is the current shortest remaining travel-time cost. The routing fractions are

\[
p(\ell,k,h,t)
=
\frac{\exp\{-\theta[C(\ell,k,h,t)-C^{\min}(\ell,h,t)]\}}
{\sum_{k'\in\mathcal K(\ell,h)}
\exp\{-\theta[C(\ell,k',h,t)-C^{\min}(\ell,h,t)]\}}.
\]

Because \(C\) is measured in seconds, \(\theta\) has units of \(\mathrm{s}^{-1}\).
