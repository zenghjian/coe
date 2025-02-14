function geoErrors = calc_geoErrors(matches, gtsrc, gttar, target)
    if isstruct(target)
        geoDist = geodesicDistance(target) ./ sqrt(sum(calc_areas(target.vertices, target.faces)));
    else
        geoDist = target;
    end
    if size(matches, 2) > 1
        matches = sortrows(matches, 1);
        assignment = matches(:, 2);
    else
        assignment = matches;
    end
    gtsrc = min(max(gtsrc, 1), 4998);
    geoErrors = geoDist(sub2ind(size(geoDist), assignment(gtsrc), gttar));
end
