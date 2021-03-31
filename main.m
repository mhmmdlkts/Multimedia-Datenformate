load penny

surf(P)
view(2)
colormap copper
shading interp
axis ij square off

Q = dct(P,[],1);
R = dct(Q,[],2);

X = R(:);

[~,ind] = sort(abs(X),'descend');
coeffs = 1;
while norm(X(ind(1:coeffs)))/norm(X) < 0.9998
   coeffs = coeffs + 1;
end
fprintf('%d of %d coefficients are sufficient\n',coeffs,numel(R))

R(abs(R) < abs(X(ind(coeffs)))) = 0;

S = idct(R,[],2);
T = idct(S,[],1);

surf(T)
view(2)
shading interp
axis ij square off