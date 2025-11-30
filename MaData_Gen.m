clc;
clear;

f1 = figure('Position',[20 400 500 375]);
f2 = figure('Position',[20 60 500 375]);

% 生成数据集名称
filename = 'test_data';

mkdir( [filename '/L']);
mkdir( [filename '/R']);

fid = fopen( [filename '/position.csv'], 'w'); % 创建目标位置文件
fprintf(fid, '%s,%s,%s,%s\n', 'index','px','py','pz');%表头信息
    
% 保存背景图像
    %左
    set(0, 'currentFigure', f1);
    axis([-50 50 -50 50 -50 50]);
    grid on;
    xticks(-50:25:50)
    yticks(-50:25:50)
    zticks(-50:25:50)
    view([-4,-5,5])
    camlight('headlight')
    file_namel = [filename '/Background_L' ];    
    print(f1, file_namel, '-dpng', '-r80') 
    
    %右
    set(0, 'currentFigure', f2);
    axis([-50 50 -50 50 -50 50]);
    grid on;
    xticks(-50:25:50)
    yticks(-50:25:50)
    zticks(-50:25:50)
    view([-5,-4,5])
    camlight('headlight')
    file_namer = [filename '/Background_R' ];
    print(f2, file_namer, '-dpng', '-r80') 
    
% 生成图像    
for step =1:1024
    px=unifrnd(-45,45);
    py=unifrnd(-45,45);
    pz=unifrnd(-45,45);
    [ex,ey,ez]  = ellipsoid(px, py, pz, 5, 5, 5,200); %小球位置p，半径5，分辨率200
    
    % 生成左视角图像
    set(0, 'currentFigure', f1);
    surf(ex,ey,ez,'LineStyle','none','FaceColor', 'b');
    axis([-50 50 -50 50 -50 50]);
    xticks(-50:25:50)
    yticks(-50:25:50)
    zticks(-50:25:50)
    view([-4,-5,5])
    camlight('headlight')
    name = sprintf('%06d', step);
    file_namel = [filename '/L/L' name ];
    print(f1, file_namel, '-dpng', '-r80') ;
    
    % 生成右视角图像
    set(0, 'currentFigure', f2);
    surf(ex,ey,ez,'LineStyle','none','FaceColor', 'b') ;
    axis([-50 50 -50 50 -50 50]);
    xticks(-50:25:50)
    yticks(-50:25:50)
    zticks(-50:25:50)
    view([-5,-4,5])
    camlight('headlight')
    file_namer = [filename '/R/R' name ];    
    print(f2, file_namer, '-dpng', '-r80') 
    
    %保存坐标信息
   fprintf(fid, '%s,%f,%f,%f\n', name,px,py,pz);
                 
end

fclose(fid);
close all;


