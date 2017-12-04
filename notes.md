- Questions you can ask:
  - Which features describe the data the most? How important is each
    feature?
  - What linear combination of features explains the variation of the
    data the most?
  - Which models seem to predict the classifications the best? Perhaps
    consider figuring out why this might be the case.
  - Separate the data based on the values of some features. Does the
    behavior/prediction change in either group? Why might it change?
    For instance, consider data which describes matches in a sport,
    where the features include which team was the home team and which
    team was the away team. You could check the importance of this
    feature:
    - For the whole dataset.
    - For games beyond a certain year.
    - For games in a single year.

    And you could take that feature importance and perhaps try to
    visualize it. How has the importance of a feature changed over
    time?

  - Does it make sense to do a "bootstrapped" feature importance? If
    I'm going to make some judgments about how important a feature is
    based on the numbers that come out of this algorithm, it might be
    nice to figure out how consistent that number is going to be with
    different samples. If it varies wildly, then I probably can't
    trust it.
