<!-- This page is for an official declaration. -->

\vspace*{\fill}
\noindent
\textit{
I, Oussama EL HAMZAOUI confirm that the work presented in this report is my own (with the help of Claude.ai). Where information has been derived from other sources, I confirm that this has been indicated in the report.
}
\vspace*{\fill}
\pagenumbering{gobble}
\newpage
<!--  -->

# Abstract {.unnumbered}

<!-- This is the abstract -->

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam et turpis gravida, lacinia ante sit amet, sollicitudin erat. Aliquam efficitur vehicula leo sed condimentum. Phasellus lobortis eros vitae rutrum egestas. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Donec at urna imperdiet, vulputate orci eu, sollicitudin leo. Donec nec dui sagittis, malesuada erat eget, vulputate tellus. Nam ullamcorper efficitur iaculis. Mauris eu vehicula nibh. In lectus turpis, tempor at felis a, egestas fermentum massa.

\pagenumbering{roman}
\setcounter{page}{1}

# Acknowledgements {.unnumbered}

<!-- This is for acknowledging all of the people who helped out -->

Interdum et malesuada fames ac ante ipsum primis in faucibus. Aliquam congue fermentum ante, semper porta nisl consectetur ut. Duis ornare sit amet dui ac faucibus. Phasellus ullamcorper leo vitae arcu ultricies cursus. Duis tristique lacus eget metus bibendum, at dapibus ante malesuada. In dictum nulla nec porta varius. Fusce et elit eget sapien fringilla maximus in sit amet dui.

Mauris eget blandit nisi, faucibus imperdiet odio. Suspendisse blandit dolor sed tellus venenatis, venenatis fringilla turpis pretium. Donec pharetra arcu vitae euismod tincidunt. Morbi ut turpis volutpat, ultrices felis non, finibus justo. Proin convallis accumsan sem ac vulputate. Sed rhoncus ipsum eu urna placerat, sed rhoncus erat facilisis. Praesent vitae vestibulum dui. Proin interdum tellus ac velit varius, sed finibus turpis placerat.

<!-- Use the \newpage command to force a new page -->

\newpage

<!-- This is for table of content -->

\pagenumbering{gobble}

\tableofcontents

\newpage

<!-- This is for the list of figures -->

\listoffigures
<!--
The \listoffigures will use short captions first, and the whole caption if none is present. To keep this list readable, ensure each figure has a short caption, e.g.
![main_text_caption](source/figures/my_image.pdf ){#fig:mylabel}{ width=50% short-caption="short caption"}

-->
\newpage


<!-- This is for the list of tables -->

\listoftables

\newpage

<!--
The \listoftables will use short captions first, and the whole caption if none is present. To keep this list readable, ensure each figure has a short caption, e.g.

+----------+----------+----------+
|   Test   |  Test2   |  Test3   |
+----------+----------+----------+
|    20    |    22    |    23    |
+----------+----------+----------+
|    34    |    35    |    36    |
+----------+----------+----------+
:  Long caption []{#tbl:tbl_ref short-caption="short caption"}

You MUST include the empty square brackets before the curly brackets.

format to use
-----------------------------------------------------------------------------------
Landmass    \%      Number of   Dolphins per    How Many    How Many    Forbidden
            stuff   Owls        Capita          Foos        Bars        Float
----------  ------  ---------   -------------   ---------   --------    -----------
North       94%     20,028      17,465          12,084      20,659      1.71
America                                                               

Benguerir   99%     65498       256,54          565656      5489        2454
-----------------------------------------------------------------------------------

: Important data for various land masses. []{short-caption="Table short caption"}


-->

<!-- This is for the list of abbreviations -->

# Abbreviations {.unnumbered}

\begin{tabbing}
\textbf{ADAS}~~~~~~~~~~~~\= \textbf{A}dvanced \textbf{D}river \textbf{A}ssistance \textbf{S}ystems \\
\textbf{CKF}    \> \textbf{C}ubature \textbf{K}alman \textbf{F}ilter \\
\textbf{EKF}    \> \textbf{E}xtended \textbf{K}alman \textbf{F}ilter \\
\textbf{GNSS}   \> \textbf{G}lobal \textbf{N}avigation \textbf{S}atellite \textbf{S}ystem \\
\textbf{HD}     \> \textbf{H}igh \textbf{D}efinition (as in HD maps) \\
\textbf{IMU}    \> \textbf{I}nertial \textbf{M}easurement \textbf{U}nit \\
\textbf{KF}     \> \textbf{K}alman \textbf{F}ilter \\
\textbf{LIDAR}  \> \textbf{L}ight \textbf{D}etection and \textbf{R}anging \\
\textbf{MCL}    \> \textbf{M}onte \textbf{C}arlo \textbf{L}ocalization \\
\textbf{PF}     \> \textbf{P}article \textbf{F}ilter \\
\textbf{RADAR}  \> \textbf{R}adio \textbf{D}etection and \textbf{R}anging \\
\textbf{RBPF}   \> \textbf{R}ao-\textbf{B}lackwellized \textbf{P}article \textbf{F}ilter \\
\textbf{SLAM}   \> \textbf{S}imultaneous \textbf{L}ocalization and \textbf{M}apping \\
\textbf{UKF}    \> \textbf{U}nscented \textbf{K}alman \textbf{F}ilter \\
\textbf{V2X}    \> \textbf{V}ehicle-to-Everything Communication \\
\end{tabbing}

\newpage

# Overview {.unnumbered}

\newpage
# Course chapters {.unnumbered}
Section to be deleted after completion of the couse

- [ ] Introduction to Localization
    - [ ] Problem statement and motivation
    - [ ] Types of localization problems
    - [ ] State estimation challenges
    - [ ] Sensor types and characteristics
    - [ ] Sources of uncertainty in robotics
- [ ] Probability Theory Foundations
    - [ ] Random variables and probability distributions
    - [ ] Bayes' theorem
    - [ ] Conditional probability
    - [ ] Markov assumption
    - [ ] Joint and marginal probabilities
    - [ ] Gaussian distributions
- [ ] Bayesian Filtering Framework
    - [ ] Recursive state estimation
    - [ ] Prediction step (motion model)
    - [ ] Update step (measurement model)
    - [ ] Chapman-Kolmogorov equation
    - [ ] Bayes filter algorithm
    - [ ] Linear vs nonlinear systems
- [ ] Kalman Filtering
    - [ ] Linear Kalman Filter
        - [ ] System model and assumptions
        - [ ] Prediction equations
        - [ ] Update equations
        - [ ] Uncertainty propagation
    - [ ] Extended Kalman Filter (EKF)
        - [ ] Linearization process
        - [ ] Jacobian matrices
        - [ ] Algorithm implementation
    - [ ] Unscented Kalman Filter (UKF)
        - [ ] Sigma points
        - [ ] Unscented transform
        - [ ] Algorithm implementation
- [ ] Particle Filtering
    - [ ] Monte Carlo methods
    - [ ] Importance sampling
    - [ ] Particle representation
    - [ ] Sequential Importance Sampling (SIS)
    - [ ] Resampling techniques
    - [ ] Sample degeneracy and impoverishment
    - [ ] Adaptive particle filtering
- [ ] Advanced Topics
    - [ ] Multi-hypothesis tracking
    - [ ] SLAM basics
    - [ ] Sensor fusion techniques
    - [ ] Loop closure
    - [ ] Global vs local localization
- [ ] MATLAB Implementation: Kalman Filter
    - [ ] Linear KF implementation
        - [ ] State prediction
        - [ ] Measurement update
        - [ ] Covariance propagation
    - [ ] EKF implementation
        - [ ] System modeling
        - [ ] Jacobian computation
        - [ ] Filter implementation
    - [ ] Visualization and analysis
    - [ ] Performance evaluation
- [ ] MATLAB Implementation: Particle Filter
    - [ ] Basic PF framework
    - [ ] Particle initialization
    - [ ] Motion model implementation
    - [ ] Measurement model
    - [ ] Weight computation
    - [ ] Resampling implementation
    - [ ] Visualization tools
    - [ ] Performance metrics
- [ ] Practical Applications
    - [ ] Vehicle localization case studies
    - [ ] Robot navigation examples
    - [ ] Integration with mapping
    - [ ] Real-world challenges
    - [ ] Best practices and optimization
- [ ] Project Work
    - [ ] Implementation exercises
    - [ ] Real dataset analysis
    - [ ] Performance comparison of different filters
    - [ ] Parameter tuning
    - [ ] Documentation and presentation

<!-- 
Would you like me to elaborate on any specific chapter or create detailed content for a particular section? -->

\newpage
# Objectives {.unnumbered}


