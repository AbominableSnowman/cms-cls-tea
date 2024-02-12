# Simulation and Analysis of Morphogen Gradient Formation in Growing Domains for Different Peclet Numbers (cms-cls-tea)

Repository for team project in Computational Modelling & Simulation MSc program.
Background: One of the many mysteries in developmental biology is how cells coordinate and differentiate into specific cell types during morphogensis. A leading theory hypothesizes that the information as to whcih cell type a given cell should differentiate into is encoded in the concentration gradient of specific proteins related to morphogensis known as morphogens. This project uses the C++ library OpenFPM [1] to simulate the diffusion process of a arbitrary morphogen to study how different computational models might fit what is observed in nature.

# Abstract
One question that has puzzled developmental biologists for centuries is how multicellular organisms
begin from a single cell and form a fully functional offspring with vastly different cell types and
a specific 3-dimensional architecture. This process requires dynamic communication between the
different cells and tissues within an organism. One such way that has been identified is through
chemical signals called morphogens. Morphogens can be dispersed throughout the tissue through
many mechanisms like diffusion, cell-based dispersion and active transport. While much research
has been devoted to understanding the process of morphogen gradient formation, one aspect that
has not received much focus is the effect of growth- how increasing the volume where morphogens
can diffuse and advection, a fluid flow induced by growth affects the dynamics of diffusion. To
this end, we quantified the effect growth has on the formation of mass steady states by simulating
three models: a simple diffusion model, a diffusion model in a growing domain modeled by the level
set method and an advection-diffusion model in a growing domain. It was found that both the
diffusion plus growth and advection-diffusion plus growth models varied significantly from simple
diffusion without growth for small ranges of P´eclet while all three models agreed within a limited
P´eclet range. For larger P´eclet values, the diffusion plus growth model agreed with simple diffusion
while advection-diffusion plus growth again varied significantly

1. P. Incardona, A. Leo, Y. Zaluzhnyi, R. Ramaswamy, and I. F. Sbalzarini, “Openfpm: A scalable open framework for particle and particle-mesh codes on parallel computers,” Computer Physics Communications, vol. 241, pp. 155–177, 2019.
