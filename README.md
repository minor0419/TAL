**時間的行動定位****(****Temporal**** ****Action**** ****Localization****)**** ****調査**
作成：2022/11/7
ネクストシステム 古林

  * スケルトンベースのアクション認識のための時空間グラフ畳み込みネットワーク
  Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
  
  ![](media/image1.png)
  
  上記のように２Dヒューマンスケルトンを時空間へ拡張して行動を検出するものです。
  パーティション戦略というもので、空間-時間領域への自然な拡張を目指しています。
  
  ![](media/image2.png)
  
  2つの異なるウェイトベクトルを持つことになり、関節間の相対的な移動などの局所的な微分特性
  をモデル化することができるらしいです。
  
  論文URL：1801.07455v2.pdf (arxiv.org)
  採択された学会：AAAI2018(アメリカ人工知能学会) 助成金による研究　香港ECS、中国センスタイム社
  論文引用数：2424
  PaperWithCodeでの順位：２位　スケルトンベース部門
  GitHubのコードのURL：https://github.com/open-mmlab/mmskeleton
  
  

  * 弱教師付き時間アクションローカリゼーションのためのバックグラウンド抑制ネットワーク
  Suppression Network for Weakly-supervised Temporal Action Localization
  
  ![](media/image3.png)
  この論文の特徴は動画をRGBとオプティカルフローに分けて扱う事です。
  行動定位の課題として背景画面の取り扱いがありますが。ここでは抑制ブランチ側にフィルタを設けて背景フレームからの活性化を抑制するよう学習するようです。このようなアーキテクチャを非対称トレーニング戦略と呼び、2ブランチの重み共有アーキテクチャを持つバックグラウンド抑制ネットワーク（BaS-Net）を提案しています。
  
  論文URL： [https://arxiv.org/abs/1911.09963](https://arxiv.org/abs/1911.09963)
  採択された学会：AAAI2020(アメリカ人工知能学会)
  論文引用数：121
  PaperWithCodeでの順位：６位　時間的行動定位部門　ActivityNet-1.3データベース
  GitHubのコードのURL：https://github.com/Pilhyeon/BaSNet-pytorch
  
  
  

  * ActionFormer：トランスフォーマーによるアクションの瞬間のローカライズ
  ActionFormer：Localizing Moments of Actions with Transformers
  
  ![](media/image4.png)
  本論文はTransformer ベースのモデルを利用し、各瞬間を分類し、アクションの境界を推定することで、
  アクションインスタンスを検出するらしいです。まずビデオクリップの特徴量(I3D)のシーケンスを抽出し、これらの各特徴量をマルチスケールトランスフォーマーにより特徴ピラミッドに変換し、特徴ピラミッドを分類器と回帰器が共有して、時間ステップごとに行動候補を生成するらしいです。
  尚GitHubの実装は動画入力ではなくTHUMOS’14の計算済みI3d特徴量で、出力はテストデータのみとなっております。
  
  論文URL： [2202.07925.pdf (arxiv.org)](https://arxiv.org/pdf/2202.07925.pdf)
  採択された学会：European Conference on Computer Vision, 2022
  論文引用数：18
  PaperWithCodeでの順位：1位　時間的行動定位部門　THUMOS’14データベース
  GitHubのコードのURL： [http://github.com/happyharrycn/actionformer_release](http://github.com/happyharrycn/actionformer_release)
  
  
  
  
  
  

  * 時間的行動検出のための提案関係ネットワーク
  Proposal Relation Network for Temporal Action Detection
  
  ![](media/image5.png)
  
  この論文はViViT: A Video Vision Transformerをバックボーンとして利用しています。
  詳細はこちら…
  [https://ai-scholar.tech/articles/transformer/ongoing](https://ai-scholar.tech/articles/transformer/ongoing)
  
  バックボーン後はデータ拡張のために時間シフトの概念が用いられているそうです。そのあと提案関係モジュールが続きます。ここでは、対象となる行動の境界を正確に特定するためが自己注視操作を導入して各提案相互の依存関係を求めます。そして最終的に提案＋分類パイプラインに従って、最終的な検出結果を生成しています。提案関連モデルからパイプラインへのつながりの説明が不明瞭に感じました。
  
  論文URL： [https://paperswithcode.com/paper/proposal-relation-network-for-temporal-action](https://paperswithcode.com/paper/proposal-relation-network-for-temporal-action)
  採択された学会：不明　アリババ・グループの支援を受けています。
  論文引用数：９
  PaperWithCodeでの順位：1位　時間的行動定位部門　ActivityNet-1.3データベース
  GitHubのコードのURL：https://github.com/wangxiang1230/SSTAP 
  
  
  
  
  
  
  * 行動検出のための全体的相互作用変圧器ネットワーク
  Holistic Interaction Transformer Network for Action Detection (HIT)
  
  この論文ではHITネットワークの提案です。ここで言うアクションとは、他の人やオブジェクト、私たちを含む、環境とのインタラクション方法に関するものです。 本論文では,多くの人間の行動に不可欠な手や**ポーズ情報**を活用するマルチモーダルな包括的インタラクショントランスフォーマーネットワーク(HIT)を提案しています。それぞれが個人、オブジェクト、手動のインタラクションを別々にモデル化します。 各サブネットワーク内では、個々の相互作用ユニットを選択的にマージするイントラモダリティアグリゲーションモジュール(IMA)が導入されました。 それぞれのモダリティから得られる特徴は、優れた融合機構(AFM)を用いて結合されます。 最後に、時間的文脈から手がかりを抽出し、キャッシュメモリを用いて発生した動作をよりよく分類します。
  
  ![](media/image6.png)
  図 1：この図は、手の特徴がいかに動作の検出に不可欠であるかを例示しています。この図では、フレーム内の人物は2人とも物体と対話しています。しかし、インスタンス検出器では、両者が相互作用している物体（緑色のボックス）を検出できず、代わりに重要でない物体（灰色の破線ボックス）が選択されてしまいます。しかし、手とその間にあるもの（黄色のボックス）をキャプチャすることで、アクター（赤色のボックス）が行っている動作について、より良いアイデアをモデルに与えることができます。「持ち上げる／拾う」（左）、「運ぶ／持つ」（右）。
  
  
  
  
  
  
  ![](media/image7.png)
  図2：HITネットワークの概要。RGBストリームの上には3次元CNNバックボーンがあり、映像の特徴を抽出するために使用されます。ポーズエンコーダーは空間変換モデルです。我々は、人物、手、物体の特徴を用いて、両方のサブネットワークから豊富な局所情報を並列に計算します。そして、グローバルな文脈との相互作用をモデル化する前に、注意深い融合モジュールを用いて学習した特徴を結合します。
  
  ![](media/image8.png)
  
  図3：インタラクションモジュールの説明図。∗ はモジュール固有の入力を、P e は A（P）の人物特徴、 または A（∗）の前にあるモジュールの出力を指します。
  
  

![](media/image9.png)
図 4: Intra-Modality Aggregator(IMA) の説明図。このように、本システムでは、1つのユニットから次のユニットへの特徴量が、まず文脈上の手がかりで補強され、次にフィルタリングされます。

  論文URL： [http://arxiv.org/abs/2210.12686v1](http://arxiv.org/abs/2210.12686v1)
  採択された学会：不明　マイクロソフトR&amp;D
  論文引用数：不明　Google Scholar 検索不能
  PaperWithCodeでの順位：1位　時間的行動定位部門　J-HMDB-21データベース
  GitHubのコードのURL：https://github.com/joslefaure/hit 　(現在何もアップロードされていません)





  * You Only Watch Once: リアルタイム時空間行動定位に向けた統一的な CNN アーキテクチャ
  You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization
  
  ![](media/image10.png)
  
  この論文では、ビデオストリーム中の時空間行動定位をリアルタイムで行うための統一的なCNNアーキテクチャであるYOWOを紹介しています。YOWOは、時間情報と空間情報を同時に抽出する2つのブランチを持つ1段のアーキテクチャであり、そのブランチから予測します。YOWOは一段構成で、時間的・空間的情報を同時に抽出し、ビデオクリップから直接バウンディングボックスと行動確率を一度の評価で予測します。YOWOは世界で初めて、かつ唯一のシングルステージのアーキテクチャです。（正直言って他のアーキテクチャは複雑じゃありませんでしたか？）YOWOアーキテクチャは、2つのブランチを持つ1段のネットワークです。1つのブランチはキーフレームの空間的特徴を抽出する。すなわちもう一方のブランチは、以前のフレームからなるクリップの時空間的特徴を3D-CNNでモデル化します。最後に、融合された特徴量を用いてフレームレベルの検出を行い、アクションチューブを生成するためのリンキングアルゴリズムを提供します。リアルタイム性を維持するため、YOWOはRGBモダリティで運用されています。しかし、YOWOはRGBモダリティで動作していることに注意する必要があります。アーキテクチャは、RGBモダリティ上でのみ動作するように制限されているわけではありません。YOWOに異なるブランチを挿入することで、異なるモダリティで動作させることができます。オプティカルフロー、深度などの異なるモダリティのために、YOWOに異なるブランチを挿入することができます。さらに、2D-CNN および 3D-CNN ブランチには、任意の CNN アーキテクチャを使用することができます。
  
  
  
  ![](media/image11.png)
  図：2D-CNNブランチと3D-CNNブランチから来る出力特徴マップを集約するためのチャネル融合とアテンション機構
  
  ![](media/image12.png)
  図：(a)3D-CNNバックボーンと(b)2D-CNNバックボーンの活性化マップ。3D-CNNバックボーンは、動き／アクションが起きている部分に着目しています。2D-CNNバックボーンは、キーフレームに写っている全ての人物に着目している。例として、バレーボールのスパイク（上）、スケートボード（中）、ロープクライミング（下）を挙げることができます。YOWOはいくつかの欠点も持っています。生成するのはキーフレームとクリップで利用可能なすべての情報に従って予測を行うため、時には誤検出をすることがあります。YOWOは正しい行動定位を行うために十分な時間的内容を必要とします。人が急に動作を始めると、初期位置の特定が難しくなります。初期状態において処理されたクリップとキーフレームにポーズ情報が含まれていないからです。そこで長期特徴量バンク（LFB: Long Term Feature Bank）を利用します。クリップの長さを長くすることで、利用可能な時間情報が増加し、結果としてYOWOの性能が向上します。LFBは時間情報を増加させる目的で活用されています。論文URL： [https://arxiv.org/pdf/1911.06644v5.pdf](https://arxiv.org/pdf/1911.06644v5.pdf)
  採択された学会：不明　ドイツ科学アカデミー（DFG）及びNVIDIA社から支援を受けている
  論文引用数：65
  PaperWithCodeでの順位：1位　時間的行動定位部門　UCF101-24データベース
  GitHubのコードのURL：[https://github.com/wei-tim/YOWO](https://github.com/wei-tim/YOWO)　　
  
  GitHubには進化版的なリポジトリが62個存在している。
  
  * Yowo_Plus
  
  ![](media/image13.png)
  
  YOWO進化版　20 Oct 2022  ·  [Jianhua Yang](https://paperswithcode.com/author/jianhua-yang)
  この技術報告では、YOWO時効検出の更新について紹介します。 我々は、3D-ResNext-101やYOLOv2を含むYOWOのオフィシャル実装と同じものを使っていますが、再実装されたYOLOv2のよりトレーニング済みの重量を使用します。 YOWO-NanoはUCF101-24で90FPSの81.0%のフレームmAPと49.7%のビデオフレームmAPを達成しました。
  
  私たちは改善するために、小さなデザイン変更をたくさん行いました。ネットワーク構造には3D-ResNext-101やYOLOv2を含むYOWOと同じものを使用しますが、実装済みのYOLOv2の事前学習重量は、YOLOv2よりも優れています。また,YOWOにおけるラベル割り当てを最適化しました。アクションインスタンスを正確に検出するために、ボックス回帰のためのGIoU損失をデプロイしました。インクリメンタルな改善の後、YOWOは公式のYOWOよりもかなり高い84.9\\%のフレームmAPと50.5\\%の動画mAPをUCF101-24で達成しました。AVAでは、最適化されたYOWOは、公式YOWOを超える16フレームの20.6\\%のフレームmAPを達成しました。32フレームのYOWOでは、RTX 3090 GPU上で25FPSの21.6フレームのmAPを実現しています。 最適化されたYOWOをYOWO-Plusと呼ぶことにしました。さらに、3D-ResNext-101を効率的な3D-ShuffleNet-v2に置き換え、軽量なアクション検出器YOWO-Nanoを設計しました。YOWO-Nano は UCF101-24 上で 90 FPS 以上の 81.0 \\% フレーム mAPと49.7\\%ビデオフレームmAPを達成します。また、AVAで約90 FPSの18.4 \\%のフレームmAPを達成しています。（要するに現状世界一）
  
  論文URL： [https://arxiv.org/pdf/2210.11219v1.pdf](https://arxiv.org/pdf/2210.11219v1.pdf)
  採択された学会：不明　
  論文引用数：なし　最近過ぎてデータなし
  PaperWithCodeでの順位： 最近過ぎてデータなし
  GitHubのコードのURL：[https://github.com/yjh0410/pytorch_yowo](https://github.com/yjh0410/pytorch_yowo)　　
  
  
