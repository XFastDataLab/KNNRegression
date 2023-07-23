%����ͼ��    
function drawshapes(points,labels)
    [m,n]=size(points);
    colors='rybgckm';
    %��״
    shapes='*o^+sx<d.^ph>dv';
    hold on
    for i=1:m
        cur_label=labels(i);        
        %colr,colg,colb�ǻ�ͼ��RGB��ɫֵ
        colorindex=mod(cur_label,7)+1;    
        %ѡ��ͼ��״
        shapeindex= mod(cur_label,14)+1;
        plot(points(i,1),points(i,2),shapes(shapeindex),'markerface','w','markeredge',colors(colorindex));
        set(gcf,'unit','normalized','position',[.4 .4 .2 .30])
        set(gca,'FontSize',15);
    end
end
