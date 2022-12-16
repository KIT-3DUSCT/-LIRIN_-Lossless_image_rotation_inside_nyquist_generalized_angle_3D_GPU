function [out]=LIRIN3DGPU(A, angle, image_origin, rotation_point, resolution)
    % Rotates lossless in the Fourier domain! Abitrary angles, abitrary
    % rotation point. Limitation: Only z-axis rotation.
    % No padding, but wrap around.
    % Parameters:
    % A: image 2d/3d
    % angle: rotation angle [degrees]
    % image_origin: x,y,z [meter]
    % rotation_point: x,y [meter]
    % resolution: resolution of pixels [meter]
    % out: rotated image 2d/3d

    out = zeros(size(A));

    width = size(A, 1);
    height = size(A, 2);
    rotation_point_x = rotation_point(1)-image_origin(1); %delta vector
    rotation_point_y = rotation_point(2)-image_origin(2);
    
    x=(0:width-1).*resolution;
    x=x-mean(x);
    y=(0:height-1).*resolution;
    y=y-mean(y);

    transform_matrix = makehgtform('translate',[rotation_point_x rotation_point_y 0])*makehgtform('axisrotate',[0 0 1],2*pi*angle/360)*makehgtform('translate',[-rotation_point_x -rotation_point_y 0]);
    pos_org=single(cat(4,repmat(x',[1, size(y,2)]),repmat(y,[size(x,2), 1])));
    
    % slicer: 3d to 2d
    for z=1:size(A, 3)
        pos_new = pos_org;
    
        for x2=1:length(x)
            for y2=1:length(y)
                pos_new(x2,y2,1,1:4)=([squeeze(pos_new(x2,y2,1,1:2))' 0 1])*transform_matrix';
            end
        end
    
        out_layer=GPUAmplitudeExtractionFourier2D(A(:,:,z),squeeze(pos_new(:,:,1,1)),squeeze(pos_new(:,:,1,2)),1./resolution);
        out(:,:,z)=reshape(out_layer,[length(x) length(y)]);
    end
end
