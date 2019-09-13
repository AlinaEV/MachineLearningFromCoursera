function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K = number of centroids
K = size(centroids, 1);
%number of examples
m = size(X,1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

distance = zeros(m, K);
for i = 1:K
  distance(:,i) = sum((bsxfun(@minus, X, centroids(i,:))).^2,2); %[300x2][3x2]=[300x2]; sum(,2)-over columns
endfor
for i = 1:m
  [j idx(i)] = min(distance(i,:));
endfor

% =============================================================

end

