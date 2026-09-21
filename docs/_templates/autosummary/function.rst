{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{% if module + "." + objname in gallery_backreferences %}
.. minigallery:: {{ module }}.{{ objname }}
   :add-heading: Examples using ``{{ objname }}``
{% endif %}
