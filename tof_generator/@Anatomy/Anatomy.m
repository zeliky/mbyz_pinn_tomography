classdef Anatomy < handle
    properties (SetAccess = private)
        %% Coordinate Properties %%
        ra; rb; ang; x0; y0;
        %% Color Properties %%
        R; G; B;
        %% Value Properties %%
        value;
        %% Grid Properties %%
        m, n; x; y; z; margin;  
    end
    methods
        function setProperties(this)
            set(0,'DefaultAxesFontSize', 14)
            set(0,'DefaultTextFontSize', 14)
            %% Coordinate Properties %%
            this.ra=1; this.rb=1; this.ang=0; this.x0=5; this.y0=5;
            %% Color Properties %%
            this.R=0.5; this.G=0.5; this.B=0.5;
            %% Value Properties %%
            this.value = 2;
            %% Grid Properties %%
            this.m = 60;
            this.n = 60;
            this.x = linspace(1,this.m,this.m);
            this.y = linspace(1,this.n,this.n);
            this.z=1;
        end
        function setCoordinates(this, ra, rb, ang)
            this.ra = ra; this.rb = rb; this.ang = ang;
        end
        function setCenter(this, x0, y0)
            this.x0 = x0; this.y0 = y0;
        end
        function setColor(this, R, G, B)      
            this.R = R; this.G = G; this.B = B;
        end
        function setValue(this, value)      
            this.value = value;
        end
        function [V] = addTiltedEllipse(this, V)
            [ex, ey] = ellipse(this.ra, this.rb, this.ang, this.x0, this.y0);
            x = this.x; y = this.y;
            inPoints = this.polygrid(ex, ey);
            u = inPoints(:, 1); v = inPoints(:, 2);
            for i = 1:length(u)
                k = find(x==u(i));
                l = find(y==v(i));
                V(k,l)=this.value;
            end
%             fill(ex, ey, [this.R,this.G,this.B], 'EdgeColor', [this.R,this.G,this.B])
        end
        function chooseGridSize(this, m, n)
            this.m = m;
            this.n = n;
            this.x = linspace(1,this.m,this.m);
            this.y = linspace(1,this.n,this.n);
            this.z=1;
        end
        function [inPoints] = polygrid(this, xv, yv)
        % https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/41454/versions/9/previews/polygrid.m/index.html
        % https://www.mathworks.com/matlabcentral/fileexchange/41454-grid-of-points-within-a-polygon
        x = this.x; y = this.y;
        %Find the bounding rectangle
            lower_x = 1; %min(xv);
            higher_x = max(xv);

            lower_y = 1; %min(yv);
            higher_y = max(yv);
        %Create a grid of points within the bounding rectangle
            inc_x = x(2)-x(1); %1/N;
            inc_y = y(2)-y(1); %1/N;

            interval_x = lower_x:inc_x:higher_x;
            interval_y = lower_y:inc_y:higher_y;

            [bigGridX, bigGridY] = meshgrid(interval_x, interval_y);

        %Filter grid to get only points in polygon
            in = inpolygon(bigGridX(:), bigGridY(:), xv, yv);
        %Return the co-ordinates of the points that are in the polygon
            inPoints = [bigGridX(in), bigGridY(in)];
        end
    end
end
function [x y] =ellipse(ra,rb,ang,x0,y0,C,Nb)
    % function h=ellipse(ra,rb,ang,x0,y0,C,Nb)
    % Ellipse adds ellipses to the current plot
    %
    % ELLIPSE(ra,rb,ang,x0,y0) adds an ellipse with semimajor axis of ra,
    % a semimajor axis of radius rb, a semimajor axis of ang, centered at
    % the point x0,y0.
    %
    % The length of ra, rb, and ang should be the same.
    % If ra is a vector of length L and x0,y0 scalars, L ellipses
    % are added at point x0,y0.
    % If ra is a scalar and x0,y0 vectors of length M, M ellipse are with the same
    % radii are added at the points x0,y0.
    % If ra, x0, y0 are vectors of the same length L=M, M ellipses are added.
    % If ra is a vector of length L and x0, y0 are  vectors of length
    % M~=L, L*M ellipses are added, at each point x0,y0, L ellipses of radius ra.
    %
    % ELLIPSE(ra,rb,ang,x0,y0,C)
    % adds ellipses of color C. C may be a string ('r','b',...) or the RGB value.
    % If no color is specified, it makes automatic use of the colors specified by
    % the axes ColorOrder property. For several circles C may be a vector.
    %
    % ELLIPSE(ra,rb,ang,x0,y0,C,Nb), Nb specifies the number of points
    % used to draw the ellipse. The default value is 300. Nb may be used
    % for each ellipse individually.
    %
    % h=ELLIPSE(...) returns the handles to the ellipses.
    %
    % usage exmple: the following produces a red ellipse centered at 1,1
    % and tipped down at a 45 deg axis from the x axis
    % ellipse(1,2,pi/4,1,1,'r')
    %
    % note that if ra=rb, ELLIPSE plots a circle
    %

    % written by D.G. Long, Brigham Young University, based on the
    % CIRCLES.m original
    % written by Peter Blattner, Institute of Microtechnology, University of
    % Neuchatel, Switzerland, blattner@imt.unine.ch

    % Check the number of input arguments

    if nargin<1,
        ra=[];
    end;
    if nargin<2,
        rb=[];
    end;
    if nargin<3,
        ang=[];
    end;

    if nargin<5,
        x0=[];
        y0=[];
    end;

    if nargin<6,
        C=[];
    end

    if nargin<7,
        Nb=[];
    end

    % set up the default values

    if isempty(ra),ra=1;end;
    if isempty(rb),rb=1;end;
    if isempty(ang),ang=0;end;
    if isempty(x0),x0=0;end;
    if isempty(y0),y0=0;end;
    if isempty(Nb),Nb=300;end;
    if isempty(C),C=get(gca,'colororder');end;

    % work on the variable sizes

    x0=x0(:);
    y0=y0(:);
    ra=ra(:);
    rb=rb(:);
    ang=ang(:);
    Nb=Nb(:);

    if isstr(C),C=C(:);end;

    if length(ra)~=length(rb),
        error('length(ra)~=length(rb)');
    end;
    if length(x0)~=length(y0),
        error('length(x0)~=length(y0)');
    end;

    % how many inscribed elllipses are plotted

    if length(ra)~=length(x0)
        maxk=length(ra)*length(x0);
    else
        maxk=length(ra);
    end;

    % drawing loop

        for k=1:maxk

            if length(x0)==1
                xpos=x0;
                ypos=y0;
                radm=ra(k);
                radn=rb(k);
                if length(ang)==1
                    an=ang;
                else
                    an=ang(k);
                end;
            elseif length(ra)==1
                xpos=x0(k);
                ypos=y0(k);
                radm=ra;
                radn=rb;
                an=ang;
            elseif length(x0)==length(ra)
                xpos=x0(k);
                ypos=y0(k);
                radm=ra(k);
                radn=rb(k);
                an=ang(k);
            else
                rada=ra(fix((k-1)/size(x0,1))+1);
                radb=rb(fix((k-1)/size(x0,1))+1);
                an=ang(fix((k-1)/size(x0,1))+1);
                xpos=x0(rem(k-1,size(x0,1))+1);
                ypos=y0(rem(k-1,size(y0,1))+1);
            end;

            co=cos(an);
            si=sin(an);
            the=linspace(0,2*pi,Nb(rem(k-1,size(Nb,1))+1,:)+1);
            x=radm*cos(the)*co-si*radn*sin(the)+xpos;
            y=radm*cos(the)*si+co*radn*sin(the)+ypos;
            %   p=line(radm*cos(the)*co-si*radn*sin(the)+xpos,radm*cos(the)*si+co*radn*sin(the)+ypos);
            %   set(p,'color',C(rem(k-1,size(C,1))+1,:));

            if nargout > 0
                %     h(k)=p;
            end

        end
end