# Foursquare Location Matching

## Kaggle Notebooks

[Exploratory Data Analysis](https://www.kaggle.com/code/sugataghosh/foursquare-location-matching-eda)

[Baseline Modeling](https://www.kaggle.com/code/sugataghosh/foursquare-location-matching-baseline-modeling)

[K-Nearest Neighbors](https://www.kaggle.com/code/sugataghosh/foursquare-location-matching-knn)

[XGBoost](https://www.kaggle.com/code/sugataghosh/foursquare-location-matching-xgboost)

## Point of Interest (POI)

A [point of interest](https://en.wikipedia.org/wiki/Point_of_interest) (POI) is a specific point location that someone may find useful or interesting. An example is a point on the Earth representing the location of the Eiffel Tower, or a point on Mars representing the location of its highest mountain, [Olympus Mons](https://en.wikipedia.org/wiki/Olympus_Mons). Most consumers use the term when referring to hotels, campsites, fuel stations or any other categories used in modern automotive navigation systems. Users of a mobile device can be provided with geolocation and time aware POI service that recommends geolocations nearby and with a temporal relevance (e.g. POI to special services in a ski resort are available only in winter). The notion of POI is widely used in cartography, especially in electronic variants including GIS, and GPS navigation software.

## The Problem of POI Matching

It is useful to combine POI data obtained from multiple sources for effective reusability. One issue in merging such data is that different dataset may have variations in POI name, address, and other identifying information for the same POI. It is thus important to identify observations which refer to the same POI. The process of POI matching involves finding POI pairs that refer to the same real-world entity, which is the core issue in geospatial data integration and is perhaps the most technically difficult part of multi-source POI fusion. The raw location data can contain noise, unstructured information, and incomplete or inaccurate attributes, which makes the task even more difficult. Nonetheless, to maintain the highest level of accuracy, the data must be matched and duplicate POIs must be identified and merged with timely updates from multiple sources. A combination of machine-learning algorithms and rigorous human validation methods are optimal for effective de-duplication of such data.

## About Foursquare

[Foursquare Labs Inc.](https://foursquare.com/), commonly known as Foursquare, is an American location technology company and data cloud platform. The company's location platform is the foundation of several business and consumer products, including the [Foursquare City Guide](https://en.wikipedia.org/wiki/Foursquare_City_Guide) and [Foursquare Swarm](https://en.wikipedia.org/wiki/Foursquare_Swarm) apps. Foursquare's products include Pilgrim SDK, Places, Visits, Attribution, Audience, Proximity, and Unfolded Studio. It is one of the leading independent providers of global POI data and is dedicated to building meaningful bridges between digital spaces and physical places. Trusted by leading enterprises like Apple, Microsoft, Samsung, and Uber, Foursquare's tech stack harnesses the power of places and movement to improve customer experiences and drive better business outcomes.

## Data

**Source:** https://www.kaggle.com/competitions/foursquare-location-matching/data

The data considered in the competition comprises over one-and-a-half million place entries for hundreds of thousands of commercial Points-of-Interest (POIs) around the globe. Though the data entries may represent or resemble entries for real places, they may be contaminated with artificial information or additional noise.

The training data comprises eleven attribute fields for over one million place entries, together with:
- `id` : A unique identifier for each entry.
- `point_of_interest` : An identifier for the POI the entry represents. There may be one or many entries describing the same POI. Two entries *match* when they describe a common POI.

The pairs data is a pregenerated set of pairs of place entries from the training data designed to improve detection of matches. It includes:
- `match` : Boolean variables denoting whether or not the pair of entries describes a common POI.

The test data comprises a set of place entries with their recorded attribute fields, similar to the training set. The POIs in the test data are distinct from the POIs in the training data.

## Project Objective

The goal of the project is to match POIs together. Using the provided dataset of over one-and-a-half million places entries, heavily altered to include noise, duplications, extraneous, or incorrect information, the objective is to produce an algorithm that predicts which place entries represent the same POI. Each place entry in the data includes useful attributes like name, street address, and coordinates. Efficient and successful matching of POIs will make it easier to identify where new stores or businesses would benefit people the most.

## Evaluation Metric

**[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index).** Also known as *Jaccard similarity coefficient*, it is a statistic used for gauging the similarity and diversity of sample sets. It was developed by [Grove Karl Gilbert](https://en.wikipedia.org/wiki/Grove_Karl_Gilbert) in 1884 as his *ratio of verification (v)* and now is frequently referred to as the *Critical Success Index* in meteorology. It was later developed independently by [Paul Jaccard](https://en.wikipedia.org/wiki/Paul_Jaccard), originally giving the French name *coefficient de communauté* and independently formulated again by T. T. Tanimoto. Thus, the *Tanimoto index* or *Tanimoto coefficient* are also used in some fields. However, they are identical in generally taking the ratio of Intersection over Union. The Jaccard coefficient measures similarity between finite sample sets, and is defined as the size of the intersection divided by the size of the union of the sample sets:

$$ J(A, B) := \frac{\left\vert A \cap B \right\vert}{\left\vert A \cup B \right\vert} = \frac{\left\vert A \cap B \right\vert}{\left\vert A \right\vert + \left\vert B \right\vert - \left\vert A \cap B \right\vert}. $$

Note that by design, $0\leq J\left(A, B\right)\leq 1$. If $A$ and $B$ are both empty, define $J(A, B) = 1$. The Jaccard coefficient is widely used in computer science, ecology, genomics, and other sciences, where binary or binarized data are used. Both the exact solution and approximation methods are available for hypothesis testing with the Jaccard coefficient. See [this paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3118-5) ([arxiv version](https://arxiv.org/abs/1903.11372)) for details.

Let us assume that for a specific `id` $a$, our algorithm produces three matches $a$, $b$ and $c$ whereas the true matches are $a$, $b$, $d$ and $e$. Then the Jaccard index for the prediction on this particular `id` will be

$$ \frac{\left\vert \left\\{a, b, c\right\\} \cap \left\\{a, b, d, e\right\\} \right\vert}{\left\vert \left\\{a, b, c\right\\} \cup \left\\{a, b, d, e\right\\} \right\vert} = \frac{\left\vert \left\\{a, b\right\\} \right\vert}{\left\vert \left\\{a, b, c, d, e\right\\} \right\vert} = \frac{2}{5}. $$

Thus, while correct matching predictions are rewarded, incorrect matching predictions are penalised by equal measure. The evaluation metric is simply the mean of Jaccard indices for each of the test observations, i.e. if the test data comprises $n_{\text{test}}$ observations and $J_i$ denotes the Jaccard index corresponding to the $i$th test observation, $i = 1,2,\cdots,n_{\text{test}}$, then the final metric by which a model will be evaluated is:

$$ \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} J_i. $$