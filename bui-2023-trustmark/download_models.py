from trustmark import TrustMark

TrustMark(device="cpu", model_type="Q", use_ECC=True,
          encoding_type=TrustMark.Encoding.BCH_5, verbose=False,
          loadRemover=False, loadBBoxDetector=False)
print("Official Q encoder, decoder, and configuration are available")
