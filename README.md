# 帰納法支援アプリケーション

課題番号: 06-04

このアプリケーションは非演繹推論である帰納を支援します。
与えられた入力文を満たす帰納した文を生成し出力します。
現在対応しているデータベースは日本語WordNetのみです。

## 使いかた

入力文は標準入力であたえ、`--max_step`引数で何層まで上位概念を探索するかをあたえます。

```
$ cat data/text1.txt
フランス人はチーズを作ります
日本人はラーメンを食べます
$ cat data/text1.txt | python query.py --max_step 3
市民はチーズを作ります
市民はラーメンを食べます
$ cat data/text2.txt
猿は柿をとっていました
赤ずきんは森に向かいました
$ cat data/text2.txt | python query.py --max_step 2
猿は本質をとっていました
赤ずきんは本質に向かいました
```

## セットアップ

1. [日本語WordNet](http://compling.hss.ntu.edu.sg/wnja/)からsqlite3形式のデータをダウンロードし、`wnjpn.db`をコードと同じディレクトリに配置してください。
2. `requirements.txt`に記載されているライブラリをインストールしてください。
   `pip install -r requirements.txt`

## 現実装の制約と今後

- 「〜は〜」という日本語の単文のみ対応しています。現状英文は対応していません。
  これは日本語の分かち書きを採用しているためで、言語ごとに切り換え英文も使えるようにしたい。
- 対応する文法のバリエーションを増やし、複雑な文でも受け入れられるようにしたい。
  ニ格ヲ格など。

## 参考

- [日本語 WordNet](http://compling.hss.ntu.edu.sg/wnja/)
- [日本語WordNetのデータベースを探索するフロントエンドプログラム](http://subtech.g.hatena.ne.jp/y_yanbe/20090314/p2)
