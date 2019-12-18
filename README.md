<<<<<<< HEAD
Marching Tetrahera for GPGPU
============================
#1. 마칭 기법에 대하여
---------------------------
  마칭 기법은 raw data로 부터 대상의 표면을 복구하는 알고리즘 중 가장 널리 쓰이는 알고리즘입니다.
  Raw Data 들은 좌표 공간 상의 격자점에서 특정한 값을 갖는 Point Clouds 데이터입니다.
  마칭 기법은 이러한 격자점들이 갖는 Volume 값을 추출 기준인 Isovalue와 비교하여
  해당 Isovalue보다 큰 값들을 갖기 시작하는 영역의 표면을 재구성 하는 알고리즘입니다.
  다음 위키피디아 링크를 참조하시면 보다 쉽게 이해하실 수 있습니다.
  [Marching Square](https://en.wikipedia.org/wiki/Marching_squares)

  마칭 큐브, 마칭 테트라헤드라와 같은 기법들은 위와 같은 방식으로 3차원에서 입체의 표면을 복구하는 기법입니다.

#2. 기존 방식과 다른 점
---------------------
  기존의 마칭 큐브는 **싱글 스레드 기반**의 설계로 인하여 GPU 병렬화를 위해 다양한 방법들이 연구되었습니다.
  하지만 대부분의 개선 알고리즘들이 연결된 형태의 삼각형 그물 구조가 아닌 삼각형 수프 형태로
  최종 출력되었고 이러한 데이터는 지오메트리 쉐이더에서 필요한 삼각형 간의 연결 정보를 제공할 수 없어 렌더링 제약이 있었습니다.
  이를 위해 후처리 과정에서 별도의 Welding 과정을 거쳐야했으나 저는 이 과정을 하나로 합쳐 1-Stage로 만들었으며
  파이프라인에서 사용할 **인덱스 구조** 를 변경하여 **Cache 기법을 이용하기 위해 메모리를 적재할 때 비용이 큰 GPU**
  에서 단순 계산을 통해 알고리즘 과정에 필요한 엘리먼트들(격자 정점, 격자 모서리, 추출 단위 다면체) 에 접근할 수 있도록 하였습니다.

  이를 통해 전체 프로세스 과정이 기존 알고리즘 대비 30% 성능 향상을 얻을 수 있었습니다.

  본 논문에서는 Marching Tetrahedra 를 통해 본 논문을 구현하였습니다.
  특징으로는 기존의 직교좌표계를 구성하는 단위 육면체를 분할하여 적용한 Marching Tetrahedra가 아닌
  체심 입방 격자(Body Centered Cubic) 좌표계를 이용하였습니다.
  BCC 패턴의 공간 샘플링은 샘플링 대상의 공간 정보를 가장 잘 보존하는 샘플링 방식이라는게 증명이 되어 있으며,
  다음 '[Tetragonal Disphenoid Honeycomb](https://en.wikipedia.org/wiki/Tetragonal_disphenoid_honeycomb)'.  이라는
  특징을 활용하기 위해 사용하였습니다.

  제 프로그램에서 가장 중요한 부분은 **Element Indexing** 기법입니다.
  기존 마칭 기법들은 각각의 격자점을 순차적으로 Index하여 배열에 그 좌표를 저장하는 방식이었습니다.
  하지만 이 방식은 Interpolation 을 할 때 해당 좌표를 인덱스에서 읽어와야하기 때문에 해당 배열을 메모리에 적재해야 합니다.
  이전에 언급한 바와 같이 GPU는 **캐시를 위해 메모리를 적재하는 과정에서 1000 사이클 이상** 소모되며
  각 스레드별로 얼마 안되는 연산을 진행하여 표면을 복구할 수 있는 마칭 기법의 특성 상 지나친 오버헤드라고 할 수 있습니다.
  그렇기 때문에 이 부분의 병목을 줄이고자 저는 표면 복구 과정에서 사용해야하는 엘리먼트들인 사면체, 사면체를 구성하는 모서리,
  모서리를 구성하는 양 끝점의 좌표에 접근하기 위한 **계층적 인덱싱 기법**을 만들어 사용했습니다.

  간단하게 설명하여 각 엘리먼트들의 무게중심 좌표를 인덱스로 사용하는 방법입니다.
  마칭 스퀘어의 예를 들자면, 추출의 단위인 정사각형은 4개의 선분과 4개의 꼭지점을 가지고 있습니다.
  (0,0), (1,0) ,(1,1), (0,1) 네개의 정점으로 구성된 정사각형은 무게중심 좌표로 (0.5, 0.5)를 가지고 있습니다.
  우리는 정사각형에서 추출해야할 선분을 구하기 위해 4개의 꼭지점의 volume test 결과를 알아야하며 해당 좌표에 접근하기 위해서
  간단하게 무게중심으로부터 네개 대각선 방향으로 (-0.5, -0.5), (0.5,-0.5), (0.5, 0.5) , (-0.5, 0.5)를 하게 되면 그 값을 구할 수 있습니다.
  Interpolation 을 할 때도 마찬가지로 무게중심이 (0.5, 0)인 모서리의 양 끝점을 구하기 위해서는 무게중심의 좌표인 (0.5, 0)에
  무게중심의 좌표 (0.5, 0) 을 각각 더하고 빼면 양 끝점의 좌표인 (0, 0)과 (1, 0)의 좌표를 얻어올 수 있습니다.

  본 논문에서의 핵심이 이 인덱싱 방법이며, 상위 엘리먼트의 식별 정보를 이용하여 하위 엘리먼트로 접근이 가능하다는 장점이 있습니다.

  이 방식을 통해 저는 메모리 참조를 최소화 시킨 마칭 기법을 구현하였습니다
  최종 결과물인 object file은 기존 마칭 기법 대비 40-50%의 용량 감소를 보였으며, 연결 정보가 필요한 실시간 렌더링 기법을 적용할 수 있었습니다.
  또한 연산 시간 역시 별도의 웰딩 과정을 필요로 했던 마칭 기법에 비해 20-30% 정도 더 빠른 성능을 보였습니다.

#3. 파이프라인
------------
1. BCC 패턴으로 직교좌표계에 입력된 클라우드포인트 데이터를 샘플링합니다.
   BCC 격자점에 해당하는 좌표들은 모든 점의 좌표가 홀수 또는 짝수로만 이루어져 있는 점들입니다.
   해당 좌표의 volume 값을 isovalue와 비교하여 비교 결과를 T/F 로 저장합니다.

2. 각 BCC 격자점에서 7개 방향( 직교 좌표계 방향 3개, 공간상의 대각선 방향 4개)의 선분을 구하면
   BCC 격자를 구성하는 모든 선분들을 구할 수 있습니다. 이를 통해 양 끝점의 T/F 판별값을
   XOR 연산하여 Interpolation 이 필요한 엣지를 판별합니다

3. 각 BCC 격자점에서 (1,1,1) 방향으로의 대각선을 둘러싸는 6개의 사면체를 구하면 BCC 좌표계 공간을 모두 채울 수 있습니다.
   모든 격자점을 순회하며 6개의 사면체에 대해 삼각형 추출 결과를 얻기 위해 각 사면체 별 4개의 꼭지점의 T/F 판별 값을 읽어들여
   삼각형을 구성하는데 필요한 모서리의 인덱스를 얻어옵니다.

4. 2에서 얻은 Interpolation이 필요한 선분들에 대해 실제 Interpolation을 진행하여, 실제 좌표값을 저장합니다.
   제 논문에서는 연결된 그물 구조의 삼각형을 출력해야하므로 Edge[Idx] => {z,y,x} 형태로 저장을 합니다.

5. 3에서 구한 각 사면체별 삼각형 구성 모서리의 무게 중심 좌표를 직렬화 시킨 인덱스로 변환하고 실제 보간 좌표를 참조하는
   삼각형 인덱스를 구성합니다.

6. obj 파일로 출력합니다.

#4. 결과물
----------
*퐁 쉐이딩*
![phong shading](https://github.com/elensar92/Marching-Methods/blob/master/MarchingTetrahedra_GPU/output/PhongShading.jpg?raw=true)

*실루엣 렌더링*
![Silhouette](https://github.com/elensar92/Marching-Methods/blob/master/MarchingTetrahedra_GPU/output/SilhouetteRendering.jpg)

*KMax*
![Kmax](https://github.com/elensar92/Marching-Methods/blob/master/MarchingTetrahedra_GPU/output/dragon-kmax.PNG?raw=true)

*가우시안 커버쳐*
![Gaussian Curvature](https://github.com/elensar92/Marching-Methods/blob/master/MarchingTetrahedra_GPU/output/GaussianCurvature.jpg)

#5. Enviroment
--------------
OS : Apple MAC, Linux,(not tested on Windows)
Language : Python, OpenCL
Libraries or Frameworks : Numpy, PyOpenCL, etc

#6. Papers
=======
Marching Tetrahera for GPGPU  
============================  
#1. 마칭 기법에 대하여  
---------------------------    
  마칭 기법은 raw data로 부터 대상의 표면을 복구하는 알고리즘 중 가장 널리 쓰이는 알고리즘입니다.   
  Raw Data 들은 좌표 공간 상의 격자점에서 특정한 값을 갖는 Point Clouds 데이터입니다.  
  마칭 기법은 이러한 격자점들이 갖는 Volume 값을 추출 기준인 Isovalue와 비교하여  
  해당 Isovalue보다 큰 값들을 갖기 시작하는 영역의 표면을 재구성 하는 알고리즘입니다.  
  다음 위키피디아 링크를 참조하시면 보다 쉽게 이해하실 수 있습니다.  
  [Marching Square](https://en.wikipedia.org/wiki/Marching_squares)  
  
  마칭 큐브, 마칭 테트라헤드라와 같은 기법들은 위와 같은 방식으로 3차원에서 입체의 표면을 복구하는 기법입니다.  
  
#2. 기존 방식과 다른 점
---------------------  
  기존의 마칭 큐브는 **싱글 스레드 기반**의 설계로 인하여 GPU 병렬화를 위해 다양한 방법들이 연구되었습니다.  
  하지만 대부분의 개선 알고리즘들이 연결된 형태의 삼각형 그물 구조가 아닌 삼각형 수프 형태로  
  최종 출력되었고 이러한 데이터는 지오메트리 쉐이더에서 필요한 삼각형 간의 연결 정보를 제공할 수 없어 렌더링 제약이 있었습니다.  
  이를 위해 후처리 과정에서 별도의 Welding 과정을 거쳐야했으나 저는 이 과정을 하나로 합쳐 1-Stage로 만들었으며  
  파이프라인에서 사용할 **인덱스 구조** 를 변경하여 **Cache 기법을 이용하기 위해 메모리를 적재할 때 비용이 큰 GPU**  
  에서 단순 계산을 통해 알고리즘 과정에 필요한 엘리먼트들(격자 정점, 격자 모서리, 추출 단위 다면체) 에 접근할 수 있도록 하였습니다.  
  
  이를 통해 전체 프로세스 과정이 기존 알고리즘 대비 30% 성능 향상을 얻을 수 있었습니다.  
  
  본 논문에서는 Marching Tetrahedra 를 통해 본 논문을 구현하였습니다.  
  특징으로는 기존의 직교좌표계를 구성하는 단위 육면체를 분할하여 적용한 Marching Tetrahedra가 아닌  
  체심 입방 격자(Body Centered Cubic) 좌표계를 이용하였습니다.  
  BCC 패턴의 공간 샘플링은 샘플링 대상의 공간 정보를 가장 잘 보존하는 샘플링 방식이라는게 증명이 되어 있으며,  
  다음 '[Tetragonal Disphenoid Honeycomb](https://en.wikipedia.org/wiki/Tetragonal_disphenoid_honeycomb)'.  이라는  
  특징을 활용하기 위해 사용하였습니다.  
  
  제 프로그램에서 가장 중요한 부분은 **Element Indexing** 기법입니다.  
  기존 마칭 기법들은 각각의 격자점을 순차적으로 Index하여 배열에 그 좌표를 저장하는 방식이었습니다.  
  하지만 이 방식은 Interpolation 을 할 때 해당 좌표를 인덱스에서 읽어와야하기 때문에 해당 배열을 메모리에 적재해야 합니다.
  이전에 언급한 바와 같이 GPU는 **캐시를 위해 메모리를 적재하는 과정에서 1000 사이클 이상** 소모되며  
  각 스레드별로 얼마 안되는 연산을 진행하여 표면을 복구할 수 있는 마칭 기법의 특성 상 지나친 오버헤드라고 할 수 있습니다.  
  그렇기 때문에 이 부분의 병목을 줄이고자 저는 표면 복구 과정에서 사용해야하는 엘리먼트들인 사면체, 사면체를 구성하는 모서리,  
  모서리를 구성하는 양 끝점의 좌표에 접근하기 위한 **계층적 인덱싱 기법**을 만들어 사용했습니다.  
  
  간단하게 설명하여 각 엘리먼트들의 무게중심 좌표를 인덱스로 사용하는 방법입니다.  
  마칭 스퀘어의 예를 들자면, 추출의 단위인 정사각형은 4개의 선분과 4개의 꼭지점을 가지고 있습니다.  
  (0,0), (1,0) ,(1,1), (0,1) 네개의 정점으로 구성된 정사각형은 무게중심 좌표로 (0.5, 0.5)를 가지고 있습니다.  
  우리는 정사각형에서 추출해야할 선분을 구하기 위해 4개의 꼭지점의 volume test 결과를 알아야하며 해당 좌표에 접근하기 위해서  
  간단하게 무게중심으로부터 네개 대각선 방향으로 (-0.5, -0.5), (0.5,-0.5), (0.5, 0.5) , (-0.5, 0.5)를 하게 되면 그 값을 구할 수 있습니다.  
  Interpolation 을 할 때도 마찬가지로 무게중심이 (0.5, 0)인 모서리의 양 끝점을 구하기 위해서는 무게중심의 좌표인 (0.5, 0)에  
  무게중심의 좌표 (0.5, 0) 을 각각 더하고 빼면 양 끝점의 좌표인 (0, 0)과 (1, 0)의 좌표를 얻어올 수 있습니다.  
  
  본 논문에서의 핵심이 이 인덱싱 방법이며, 상위 엘리먼트의 식별 정보를 이용하여 하위 엘리먼트로 접근이 가능하다는 장점이 있습니다.  
  
  이 방식을 통해 저는 메모리 참조를 최소화 시킨 마칭 기법을 구현하였습니다  
  최종 결과물인 object file은 기존 마칭 기법 대비 40-50%의 용량 감소를 보였으며, 연결 정보가 필요한 실시간 렌더링 기법을 적용할 수 있었습니다.  
  또한 연산 시간 역시 별도의 웰딩 과정을 필요로 했던 마칭 기법에 비해 20-30% 정도 더 빠른 성능을 보였습니다.  
  
#3. 파이프라인
------------
1. BCC 패턴으로 직교좌표계에 입력된 클라우드포인트 데이터를 샘플링합니다.  
   BCC 격자점에 해당하는 좌표들은 모든 점의 좌표가 홀수 또는 짝수로만 이루어져 있는 점들입니다.  
   해당 좌표의 volume 값을 isovalue와 비교하여 비교 결과를 T/F 로 저장합니다.  

2. 각 BCC 격자점에서 7개 방향( 직교 좌표계 방향 3개, 공간상의 대각선 방향 4개)의 선분을 구하면  
   BCC 격자를 구성하는 모든 선분들을 구할 수 있습니다. 이를 통해 양 끝점의 T/F 판별값을  
   XOR 연산하여 Interpolation 이 필요한 엣지를 판별합니다  

3. 각 BCC 격자점에서 (1,1,1) 방향으로의 대각선을 둘러싸는 6개의 사면체를 구하면 BCC 좌표계 공간을 모두 채울 수 있습니다.  
   모든 격자점을 순회하며 6개의 사면체에 대해 삼각형 추출 결과를 얻기 위해 각 사면체 별 4개의 꼭지점의 T/F 판별 값을 읽어들여  
   삼각형을 구성하는데 필요한 모서리의 인덱스를 얻어옵니다.
   
4. 2에서 얻은 Interpolation이 필요한 선분들에 대해 실제 Interpolation을 진행하여, 실제 좌표값을 저장합니다.  
   제 논문에서는 연결된 그물 구조의 삼각형을 출력해야하므로 Edge[Idx] => {z,y,x} 형태로 저장을 합니다.  

5. 3에서 구한 각 사면체별 삼각형 구성 모서리의 무게 중심 좌표를 직렬화 시킨 인덱스로 변환하고 실제 보간 좌표를 참조하는
   삼각형 인덱스를 구성합니다.  

6. obj 파일로 출력합니다.

#4. 결과물
----------
*퐁 쉐이딩*  
![phong shading](https://github.com/elensar92/Marching-Methods/blob/master/MarchingTetrahedra_GPU/output/PhongShading.jpg?raw=true)  
  
*실루엣 렌더링*  
![Silhouette](https://github.com/elensar92/Marching-Methods/blob/master/MarchingTetrahedra_GPU/output/SilhouetteRendering.jpg)  
  
*KMax*  
![Kmax](https://github.com/elensar92/Marching-Methods/blob/master/MarchingTetrahedra_GPU/output/dragon-kmax.PNG?raw=true)  
  
*가우시안 커버쳐*  
![Gaussian Curvature](https://github.com/elensar92/Marching-Methods/blob/master/MarchingTetrahedra_GPU/output/GaussianCurvature.jpg)  
  
#5. Enviroment  
--------------
OS : Apple MAC, Linux,(not tested on Windows)  
Language : Python, OpenCL  
Libraries or Frameworks : Numpy, PyOpenCL, etc  

#6. Papers  
>>>>>>> ed4b81c2b3866b9742128d2629ea3443ffc0bded
----------
[Mesh-based Marching Tetrahedra on BCC Datasets](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07299387)
[Mesh-based Marching Cube](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE06749207&language=ko_KR)
[Mesh-based Marching Cube on GPU](http://journal.cg-korea.org/archive/view_article?pid=jkcgs-24-1-1)



