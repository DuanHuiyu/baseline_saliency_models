function x = s(q)
% S(Q) Scalar part of a full quaternion.

% Copyright ? 2005 Stephen J. Sangwine and Nicolas Le Bihan.
% See the file : Copyright.m for further details.

% error(nargchk(1, 1, nargin)), error(nargoutchk(0, 1, nargout))

% This function is written in terms of its private counterpart
% so that the underlying representation of a quaternion is a
% private aspect of the class.

x = ess(q);

% $Id: s.m,v 1.2 2009/02/08 18:35:21 sangwine Exp $

