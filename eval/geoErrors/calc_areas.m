function area = calc_areas(V, F)

getDiff  = @(a,b)V(F(:,a),:) - V(F(:,b),:);
getTriArea  = @(X,Y).5*sqrt(sum(cross(X,Y).^2,2));
area = getTriArea(getDiff(1,2),getDiff(1,3));

end
