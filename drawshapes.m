%画出图形    
function drawshapes(points,labels)
    [m,n]=size(points);
    colors='rybgckm';
    %形状
    shapes='*o^+sx<d.^ph>dv';
    hold on
    for i=1:m
        cur_label=labels(i);        
        %colr,colg,colb是画图的RGB颜色值
        colorindex=mod(cur_label,7)+1;    
        %选画图形状
        shapeindex= mod(cur_label,14)+1;
        plot(points(i,1),points(i,2),shapes(shapeindex),'markerface','w','markeredge',colors(colorindex));
        set(gcf,'unit','normalized','position',[.4 .4 .2 .30])
        set(gca,'FontSize',15);
    end
end
