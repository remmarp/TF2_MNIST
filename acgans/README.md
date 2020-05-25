# Simple ACGANs with Wasserstein gradient penalty.

## Generate samples
Original | Generated
------------ | -------------
![alt text][img1] | ![alt text][img2]

## Classification accuracy
Training | Validation | Test
:-------------: | :-------------: | :-------------:
99.937 % | 97.824 % | 99.002 %


## Generate conditional samples
0 | 1 | 2 | 3 | 4
:------------: | :-------------: | :-------------: | :-------------: | :-------------:
![alt text][img3] | ![alt text][img4] | ![alt text][img5] | ![alt text][img6] | ![alt text][img7] | 
5 | 6 | 7 | 8 | 9
![alt text][img8] | ![alt text][img9] | ![alt text][img10] | ![alt text][img11] | ![alt text][img12]

[img1]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_original.png "Original"
[img2]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_generated.png "ACGANs generated"
[img3]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c0_generated.png "ACGANs 0"
[img4]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c1_generated.png "ACGANs 1"
[img5]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c2_generated.png "ACGANs 2"
[img6]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c3_generated.png "ACGANs 3"
[img7]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c4_generated.png "ACGANs 4"
[img8]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c5_generated.png "ACGANs 5"
[img9]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c6_generated.png "ACGANs 6"
[img10]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c7_generated.png "ACGANs 7"
[img11]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c8_generated.png "ACGANs 8"
[img12]: https://github.com/remmarp/TF2_MNIST/blob/master/acgans/assets/acgans_c9_generated.png "ACGANs 9"
